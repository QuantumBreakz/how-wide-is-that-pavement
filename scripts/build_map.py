"""Build a self-contained interactive map of every measured segment."""
from __future__ import annotations

import json
import pathlib
import sys
import warnings

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
warnings.filterwarnings("ignore")
import geopandas as gpd

from sidewalk import config as C

OUT = pathlib.Path("outputs")


def build(path=OUT / "map/sidewalk_map.html"):
    path.parent.mkdir(parents=True, exist_ok=True)
    seg = gpd.read_file(OUT / "geo/segments_wgs84.geojson")
    axes = gpd.read_file(OUT / "geo/axes.geojson").to_crs(4326)

    keep = ["segment_id", "road", "side", "s_start", "s_end", "length_m", "width_median",
            "width_p10", "width_p90", "sigma_m", "ci_low", "ci_high", "band", "band_range",
            "band_borderline", "confidence", "artefact_risk", "artefact_reason",
            "back_edge_mode", "pct_kerb_normal", "osm_agree", "effective_width_median",
            "min_clear_width", "pct_obstructed", "n_effective", "pct_closure_ambiguous"]
    keep = [k for k in keep if k in seg.columns]
    g = seg[keep + ["geometry"]].copy()
    for c in keep:
        if g[c].dtype.kind == "f":
            g[c] = g[c].round(3)
    data = json.loads(g.to_json())
    axes_data = json.loads(axes.to_json())

    stats = dict(
        roads=int(seg.road.nunique()),
        segments=int(len(seg)),
        length_km=round(float(seg.length_m.sum()) / 1000, 2),
        median_sigma=round(float(seg.sigma_m.median()), 2),
        pct_below_min=round(100 * float(seg.loc[seg.band == "below minimum", "length_m"].sum()
                                        / seg.length_m.sum()), 1),
        high_conf=round(100 * float((seg.confidence >= 0.7).mean()), 1),
    )
    colours = {k: v for k, v in C.BAND_COLOURS.items()}

    html = TEMPLATE.replace("__DATA__", json.dumps(data)) \
                   .replace("__AXES__", json.dumps(axes_data)) \
                   .replace("__COLOURS__", json.dumps(colours)) \
                   .replace("__STATS__", json.dumps(stats))
    path.write_text(html)
    print(f"wrote {path}  ({path.stat().st_size/1024:.0f} KB, {len(seg)} segments)")
    return path


TEMPLATE = r"""<title>Tower Hamlets Pavement Widths</title>
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.css"/>
<script src="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.js"></script>
<style>
  :root{--bg:#fbfaf8;--panel:#fff;--ink:#1d1f21;--muted:#6b7177;--line:#e3e0da;}
  @media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#14161a;--panel:#1c1f24;--ink:#e9eaec;--muted:#9aa0a6;--line:#2c3036;}}
  :root[data-theme="dark"]{--bg:#14161a;--panel:#1c1f24;--ink:#e9eaec;--muted:#9aa0a6;--line:#2c3036;}
  html,body{height:100%}
  body{background:var(--bg);color:var(--ink);font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;}
  #wrap{display:flex;height:100vh;}
  #map{flex:1;min-width:0;}
  #side{width:330px;border-left:1px solid var(--line);background:var(--panel);overflow:auto;padding:16px 18px;}
  h1{font-size:16px;margin:0 0 2px;letter-spacing:-.01em}
  .sub{color:var(--muted);font-size:12px;margin-bottom:14px}
  .kpis{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-bottom:16px}
  .kpi{background:var(--bg);border:1px solid var(--line);border-radius:8px;padding:8px 10px}
  .kpi b{display:block;font-size:17px;font-variant-numeric:tabular-nums}
  .kpi span{color:var(--muted);font-size:11px}
  .legend div{display:flex;align-items:center;gap:8px;margin:3px 0;font-size:12px}
  .sw{width:14px;height:14px;border-radius:3px;flex:none}
  table{width:100%;border-collapse:collapse;font-size:12px;margin-top:6px}
  td{padding:3px 0;vertical-align:top}
  td:first-child{color:var(--muted);padding-right:10px;white-space:nowrap}
  .pill{display:inline-block;padding:1px 7px;border-radius:99px;font-size:11px;color:#fff}
  .warn{color:#c2410c}
  .muted{color:var(--muted);font-size:12px}
  select{width:100%;padding:6px;border-radius:6px;border:1px solid var(--line);background:var(--panel);color:var(--ink);margin-bottom:12px}
  @media(max-width:760px){#wrap{flex-direction:column}#side{width:auto;border-left:0;border-top:1px solid var(--line);max-height:46vh}}
</style>
<div id="wrap">
  <div id="map"></div>
  <div id="side">
    <h1>Pavement widths</h1>
    <div class="sub">Tower Hamlets &amp; Spitalfields &mdash; measured from OS MasterMap</div>
    <div class="kpis" id="kpis"></div>
    <select id="roadsel"></select>
    <div class="legend" id="legend"></div>
    <div id="detail" style="margin-top:16px"><div class="muted">Click a segment for its full evidence.</div></div>
  </div>
</div>
<script>
const DATA=__DATA__, AXES=__AXES__, COLOURS=__COLOURS__, STATS=__STATS__;
const map=L.map('map',{zoomControl:true}).setView([51.518,-0.073],15);
L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png',{maxZoom:20,attribution:'&copy; OpenStreetMap contributors'}).addTo(map);
L.geoJSON(AXES,{style:{color:'#9aa0a6',weight:1,opacity:.5,dashArray:'3 4'}}).addTo(map);

const kpi=[['roads','Roads'],['segments','Segments'],['length_km','km measured'],['median_sigma','Median &sigma; (m)'],['pct_below_min','% below min'],['high_conf','% high conf.']];
document.getElementById('kpis').innerHTML=kpi.map(([k,l])=>`<div class="kpi"><b>${STATS[k]}</b><span>${l}</span></div>`).join('');
document.getElementById('legend').innerHTML=Object.entries(COLOURS).map(([b,c])=>`<div><i class="sw" style="background:${c}"></i>${b}</div>`).join('');

let selected=null;
function style(f){const p=f.properties;return{color:COLOURS[p.band]||'#999',weight:Math.max(3,Math.min(11,p.width_median*1.9)),opacity:.85};}
const layer=L.geoJSON(DATA,{style,onEachFeature:(f,l)=>{
  l.on('click',()=>{if(selected)layer.resetStyle(selected);selected=l;l.setStyle({color:'#111',weight:style(f).weight+2,opacity:1});show(f.properties);});
  l.bindTooltip(`${f.properties.road} (${f.properties.side}) &mdash; ${f.properties.width_median} m`,{sticky:true});
}}).addTo(map);
map.fitBounds(layer.getBounds(),{padding:[24,24]});

const roads=[...new Set(DATA.features.map(f=>f.properties.road))].sort();
const sel=document.getElementById('roadsel');
sel.innerHTML='<option value="">All roads</option>'+roads.map(r=>`<option>${r}</option>`).join('');
sel.onchange=()=>{const r=sel.value;layer.clearLayers();
  layer.addData({type:'FeatureCollection',features:DATA.features.filter(f=>!r||f.properties.road===r)});
  try{map.fitBounds(layer.getBounds(),{padding:[24,24]})}catch(e){}};

function row(k,v){return `<tr><td>${k}</td><td>${v}</td></tr>`}
function show(p){
  const risk=p.artefact_risk>=0.6?'warn':'';
  document.getElementById('detail').innerHTML=`
   <div style="font-weight:600">${p.road}</div>
   <div class="muted">${p.side} side &middot; ${p.s_start}&ndash;${p.s_end} m (${p.length_m} m long)</div>
   <div style="margin:10px 0 4px;font-size:26px;font-variant-numeric:tabular-nums">${p.width_median} m</div>
   <div><span class="pill" style="background:${COLOURS[p.band]}">${p.band}</span>
        ${p.band_borderline?'<span class="muted"> &nbsp;too close to call: '+p.band_range+'</span>':''}</div>
   <table>
     ${row('95% interval',`${p.ci_low} &ndash; ${p.ci_high} m (&sigma; ${p.sigma_m} m)`)}
     ${row('Range in segment',`P10 ${p.width_p10} &ndash; P90 ${p.width_p90} m`)}
     ${row('Confidence',(p.confidence??0).toFixed(2))}
     ${row('Artefact risk',`<span class="${risk}">${(p.artefact_risk??0).toFixed(2)} &mdash; ${p.artefact_reason||'none'}</span>`)}
     ${row('Kerb-normal used',`${Math.round(p.pct_kerb_normal)}% of cross-sections`)}
     ${row('Back edge',p.back_edge_mode)}
     ${row('OSM agrees footway',p.osm_agree==null?'n/a':Math.round(p.osm_agree*100)+'%')}
     ${row('Independent samples',p.n_effective)}
     ${p.min_clear_width!=null?row('Effective walkable',`${p.effective_width_median} m median, pinches to ${p.min_clear_width} m`):''}
   </table>
   <div class="muted" style="margin-top:10px">Effective walkable width is an optional accessibility derivative and is not substituted for the geometric footway width.</div>`;
}
</script>
"""

if __name__ == "__main__":
    build()
