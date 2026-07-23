"""Interactive, browser-only replay viewer for LLM evaluation debug JSONL files.

The viewer deliberately has no web framework dependency.  It is a single HTML
file, so it can be opened from an experiment artifact directory or served with
``python -m http.server``.  World tiles are reconstructed from the positions
and visible-object descriptions recorded in each agent observation; this keeps
the replay useful even when a full JAX environment state was not saved.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


_POSITION = re.compile(r"Position: \(x=(-?\d+), y=(-?\d+)\)")
_ROLE = re.compile(r"Role: ([^\n]+)")
_OBJECT = re.compile(r"^- ([\w_]+).*?\(x=(-?\d+), y=(-?\d+)\)", re.MULTILINE)
_STAT = re.compile(r"^- (health|food|drink|energy|mana|xp):\s*([\d.]+)", re.MULTILINE)


def _text(value: Any) -> str:
    return value if isinstance(value, str) else ""


def _agent_snapshot(agent_id: str, agent: dict[str, Any]) -> dict[str, Any]:
    long = _text(agent.get("obs_long_term"))
    short = _text(agent.get("obs_short_term"))
    position = _POSITION.search(long)
    role = _ROLE.search(long)
    objects = [
        {"kind": kind, "x": int(x), "y": int(y)}
        for kind, x, y in _OBJECT.findall(long)
        if kind.lower() not in {"agent", "grass"}
    ]
    return {
        "id": str(agent_id),
        "x": int(position.group(1)) if position else None,
        "y": int(position.group(2)) if position else None,
        "role": role.group(1).strip() if role else "agent",
        "stats": {name: float(value) for name, value in _STAT.findall(short)},
        "objects": objects,
        "action": agent.get("parsed_action") or "No action",
        "output": _text(agent.get("llm_raw_output")),
        "reasoning": _text(agent.get("reasoning") or agent.get("raw_reasoning")),
        "input": "\n\n".join(part for part in (long, short) if part),
        "messages_sent": agent.get("message_sent"),
        "messages_received": agent.get("messages_received") or [],
        "scratchpad": _text(agent.get("scratchpad")),
        "tokens": {
            "input": agent.get("input_tokens") or 0,
            "output": agent.get("output_tokens") or 0,
        },
        "image": agent.get("image_base64"),
    }


def build_replay_data(records: list[dict[str, Any]], summary: dict[str, Any] | None = None) -> dict[str, Any]:
    """Normalise evaluator debug records into compact data consumed by the UI."""
    steps = []
    for record in records:
        if record.get("type") == "debrief":
            continue
        agents = {
            str(agent_id): _agent_snapshot(str(agent_id), agent)
            for agent_id, agent in (record.get("agents") or {}).items()
        }
        # A bodyless leader/commander has no map position but is still an
        # active participant whose plan and response belong in the replay.
        leader = record.get("leader")
        if isinstance(leader, dict):
            prompt = leader.get("prompt_messages") or []
            agents["leader"] = {
                "id": "leader",
                "x": None,
                "y": None,
                "role": "bodyless leader",
                "stats": {},
                "objects": [],
                "action": "Plan v{} ({})".format(
                    leader.get("plan_version", "?"),
                    "valid" if leader.get("plan_valid") else "retained",
                ),
                "output": _text(leader.get("raw_output")),
                "reasoning": json.dumps(leader.get("parsed_plan"), indent=2),
                "input": "\n\n".join(
                    _text(message.get("content"))
                    for message in prompt
                    if isinstance(message, dict)
                ),
                "messages_sent": None,
                "messages_received": [],
                "scratchpad": "",
                "tokens": {},
                "image": None,
            }
        steps.append(
            {
                "step": record.get("step", len(steps)),
                "agents": agents,
                "rewards": record.get("rewards") or [],
                "dones": record.get("dones") or [],
                "routes": record.get("communication_routes") or [],
                "leader": record.get("leader"),
            }
        )
    return {"steps": steps, "summary": summary or {}}


def generate_web_replay(
    debug_jsonl_path: str | Path,
    episode_json_path: str | Path | None = None,
    output_path: str | Path | None = None,
) -> str:
    """Generate a self-contained interactive replay HTML file from debug JSONL."""
    source = Path(debug_jsonl_path)
    records = []
    for line in source.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.strip():
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    summary = None
    if episode_json_path and Path(episode_json_path).exists():
        try:
            summary = json.loads(Path(episode_json_path).read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass
    if output_path is None:
        output_path = source.with_name(source.name.replace("_debug.jsonl", "_replay.html"))
    payload = json.dumps(build_replay_data(records, summary), ensure_ascii=False).replace("</", "<\\/")
    Path(output_path).write_text(_html(source.stem, payload), encoding="utf-8")
    return str(output_path)


def _html(title: str, payload: str) -> str:
    return f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{title} replay</title>
<style>
:root{{--bg:#0b1020;--panel:#121a2e;--edge:#263657;--ink:#eaf1ff;--muted:#91a1c2;--accent:#6ee7ff;--agent0:#60a5fa;--agent1:#34d399;--agent2:#fbbf24;--agent3:#f472b6}}*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:14px system-ui,sans-serif;overflow:hidden}}button,select{{font:inherit;color:inherit;background:#1b2945;border:1px solid var(--edge);border-radius:7px;padding:7px 10px;cursor:pointer}}button:hover,button.active{{background:#234c70;border-color:var(--accent)}}header{{height:55px;display:flex;align-items:center;gap:15px;padding:0 18px;border-bottom:1px solid var(--edge);background:var(--panel)}}h1{{font-size:16px;margin:0}}.muted{{color:var(--muted)}}.tabs{{margin-left:auto;display:flex;gap:6px}}main{{height:calc(100vh - 125px)}}.tab{{display:none;height:100%}}.tab.active{{display:grid}}#replay{{grid-template-columns:minmax(0,1fr) 370px}}#world-wrap{{position:relative;overflow:hidden;background:radial-gradient(#172340 1px,transparent 1px);background-size:22px 22px}}canvas{{width:100%;height:100%;touch-action:none;cursor:grab}}canvas.dragging{{cursor:grabbing}}.hint{{position:absolute;top:14px;left:14px;padding:8px 10px;background:#0b1020c9;border:1px solid var(--edge);border-radius:7px;color:var(--muted);font-size:12px}}aside,.status-grid{{overflow:auto;background:var(--panel);border-left:1px solid var(--edge)}}#inspector{{padding:15px}}.agent-card{{border:1px solid var(--edge);border-left:4px solid var(--accent);border-radius:8px;padding:12px;margin-bottom:10px;background:#10182a}}.agent-card h2{{font-size:15px;margin:0 0 8px}}.agent-card button{{padding:3px 7px;font-size:12px;float:right}}.pill{{display:inline-block;border-radius:12px;background:#1d3152;padding:3px 8px;margin:2px;font-size:12px}}.label{{font-size:11px;text-transform:uppercase;letter-spacing:.08em;color:var(--muted);margin-top:12px}}pre{{white-space:pre-wrap;word-break:break-word;background:#0b1020;padding:9px;border-radius:6px;max-height:220px;overflow:auto;margin:4px 0;font:12px ui-monospace,monospace}}details{{margin-top:7px}}summary{{cursor:pointer;color:var(--accent)}}#status{{grid-template-columns:repeat(auto-fit,minmax(290px,1fr));gap:14px;padding:18px;border:0;background:var(--bg)}}.status-card{{background:var(--panel);border:1px solid var(--edge);border-top:4px solid var(--accent);border-radius:9px;padding:15px;height:max-content}}.status-card h2{{margin:0 0 4px;font-size:16px}}.metric{{display:flex;justify-content:space-between;border-bottom:1px solid #1c2942;padding:5px 0}}footer{{height:70px;border-top:1px solid var(--edge);background:var(--panel);display:flex;align-items:center;gap:10px;padding:10px 18px}}input[type=range]{{flex:1;accent-color:var(--accent)}}#time{{min-width:110px;color:var(--accent);font-variant-numeric:tabular-nums}}@media(max-width:850px){{#replay{{grid-template-columns:1fr;grid-template-rows:55% 45%}}aside{{border-left:0;border-top:1px solid var(--edge)}}}}
</style><body><header><h1>World Replay</h1><span class="muted" id="meta"></span><div class="tabs"><button class="active" data-tab="replay">Replay</button><button data-tab="status">Agent status</button></div></header><main><section class="tab active" id="replay"><div id="world-wrap"><canvas id="world"></canvas><div class="hint">Scroll to zoom · drag to pan · click an agent to inspect · <kbd>F</kbd> follows selection</div></div><aside id="inspector"></aside></section><section class="tab" id="status"></section></main><footer><button id="prev">◀</button><button id="play">▶ Play</button><button id="next">▶</button><select id="speed"><option value="0.25">0.25×</option><option value="0.5">0.5×</option><option value="1" selected>1×</option><option value="2">2×</option><option value="4">4×</option></select><input id="timeline" type="range" min="0" value="0"><span id="time"></span></footer>
<script>const DATA={payload};
const S=DATA.steps, colors=['#60a5fa','#34d399','#fbbf24','#f472b6','#a78bfa','#fb7185'];let i=0,selected=null,playing=false,last=0,pan={{x:0,y:0}},zoom=38,follow=false,drag=null;
const $=s=>document.querySelector(s), canvas=$('#world'),ctx=canvas.getContext('2d');$('#timeline').max=Math.max(0,S.length-1);$('#meta').textContent=`${{S.length}} ticks · ${{Object.keys(S[0]?.agents||{{}}).length}} agents`;
function esc(v){{const d=document.createElement('div');d.textContent=v??'';return d.innerHTML}}function current(){{return S[i]||{{agents:{{}}}}}}function agentColor(id){{let n=Number(id);return colors[Number.isFinite(n)?n%colors.length:colors.length-1]}}function resize(){{canvas.width=canvas.clientWidth*devicePixelRatio;canvas.height=canvas.clientHeight*devicePixelRatio;ctx.setTransform(devicePixelRatio,0,0,devicePixelRatio,0,0);draw()}}addEventListener('resize',resize);
function centre(){{let a=selected&&current().agents[selected];if(follow&&a?.x!=null)pan={{x:canvas.clientWidth/2-a.x*zoom,y:canvas.clientHeight/2-a.y*zoom}}}}function xy(x,y){{return {{x:x*zoom+pan.x,y:y*zoom+pan.y}}}}function draw(){{centre();let w=canvas.clientWidth,h=canvas.clientHeight;ctx.clearRect(0,0,w,h);let step=current(), objects=new Map();Object.values(step.agents).forEach(a=>(a.objects||[]).forEach(o=>objects.set(`${{o.x}},${{o.y}}`,o)));ctx.save();for(let x=Math.floor(-pan.x/zoom)-1;x<(w-pan.x)/zoom+1;x++)for(let y=Math.floor(-pan.y/zoom)-1;y<(h-pan.y)/zoom+1;y++){{let p=xy(x,y);ctx.strokeStyle='#1a2945';ctx.strokeRect(p.x,p.y,zoom,zoom)}}objects.forEach(o=>{{let p=xy(o.x,o.y), c={{tree:'#2f855a',stone:'#94a3b8',lava:'#ef4444',water:'#38bdf8',construction_site:'#c084fc'}}[o.kind]||'#64748b';ctx.fillStyle=c;ctx.fillRect(p.x+3,p.y+3,zoom-6,zoom-6);ctx.fillStyle='#07101d';ctx.font=`${{Math.max(9,zoom*.25)}}px system-ui`;ctx.fillText(o.kind.replace('_',' '),p.x+5,p.y+zoom*.55)}});Object.entries(step.agents).forEach(([id,a])=>{{if(a.x==null)return;let p=xy(a.x,a.y),r=Math.max(8,zoom*.32);ctx.beginPath();ctx.fillStyle=agentColor(id);ctx.arc(p.x+zoom/2,p.y+zoom/2,r,0,7);ctx.fill();if(id===selected){{ctx.strokeStyle='#fff';ctx.lineWidth=3;ctx.stroke()}}ctx.fillStyle='#fff';ctx.font=`bold ${{Math.max(10,zoom*.28)}}px system-ui`;ctx.textAlign='center';ctx.fillText('A'+id,p.x+zoom/2,p.y+zoom/2+4)}});ctx.restore()}}
function output(a){{return a.output||a.reasoning||'(no model output captured)'}}function messages(a){{let sent=a.messages_sent?.content?`Sent: ${{a.messages_sent.content}}\n`:'';let got=(a.messages_received||[]).map(m=>`From Agent ${{m.sender_idx??'?'}}: ${{m.content}}`).join('\n');return sent+got||'(no messages this tick)'}}function renderInspector(){{let step=current(), agents=Object.entries(step.agents);if(!selected||!step.agents[selected])selected=agents[0]?.[0]||null;let a=step.agents[selected];if(!a){{$ ('#inspector').innerHTML='<p class="muted">No agent data.</p>';return}}$('#inspector').innerHTML=`<div class="agent-card" style="border-left-color:${{agentColor(selected)}}"><button id="follow">${{follow?'Following':'Follow'}}</button><h2>Agent ${{selected}} · ${{esc(a.role)}}</h2><span class="pill">${{esc(a.action)}}</span><span class="pill">${{a.x??'?'}}, ${{a.y??'?'}}</span><div class="label">Vitals</div>${{Object.entries(a.stats||{{}}).map(([k,v])=>`<span class="pill">${{k}} ${{v}}</span>`).join('')||'<span class="muted">Not recorded</span>'}}<div class="label">Model output</div><pre>${{esc(output(a))}}</pre><div class="label">Communication</div><pre>${{esc(messages(a))}}</pre><details><summary>Model input / observation</summary><pre>${{esc(a.input)}}</pre></details><details><summary>Scratchpad</summary><pre>${{esc(a.scratchpad||'(empty)')}}</pre></details></div><div class="label">All agents</div>${{agents.map(([id,b])=>`<div class="agent-card" style="border-left-color:${{agentColor(id)}}"><button data-agent="${{id}}">Inspect</button><strong>Agent ${{id}}</strong> · ${{esc(b.role)}}<br><span class="muted">${{esc(b.action)}} · (${{b.x??'?'}},${{b.y??'?'}})</span></div>`).join('')}}`;$('#follow').onclick=()=>{{follow=!follow;renderInspector();draw()}};document.querySelectorAll('[data-agent]').forEach(b=>b.onclick=()=>{{selected=b.dataset.agent;follow=false;render()}})}}
function renderStatus(){{let agents=Object.entries(current().agents);$('#status').innerHTML=agents.map(([id,a])=>`<article class="status-card" style="border-top-color:${{agentColor(id)}}"><h2>Agent ${{id}} <span class="muted">${{esc(a.role)}}</span></h2><div class="muted">Position (${{a.x??'?'}}, ${{a.y??'?'}})</div><div class="label">Current action</div><strong>${{esc(a.action)}}</strong><div class="label">Vitals and usage</div>${{Object.entries(a.stats||{{}}).map(([k,v])=>`<div class="metric"><span>${{k}}</span><strong>${{v}}</strong></div>`).join('')}}<div class="metric"><span>Input tokens</span><strong>${{a.tokens.input}}</strong></div><div class="metric"><span>Output tokens</span><strong>${{a.tokens.output}}</strong></div><details><summary>Output</summary><pre>${{esc(output(a))}}</pre></details><details><summary>Messages</summary><pre>${{esc(messages(a))}}</pre></details></article>`).join('')}}
function render(){{$('#timeline').value=i;$('#time').textContent=`Tick ${{current().step??i}} / ${{S.at(-1)?.step??0}}`;renderInspector();renderStatus();draw()}}function set(n){{i=Math.max(0,Math.min(S.length-1,n));render()}}$('#timeline').oninput=e=>set(+e.target.value);$('#prev').onclick=()=>set(i-1);$('#next').onclick=()=>set(i+1);$('#play').onclick=()=>{{playing=!playing;$('#play').textContent=playing?'❚❚ Pause':'▶ Play'}};function tick(t){{if(playing&&t-last>700/+$('#speed').value){{if(i>=S.length-1)playing=false;else set(i+1);last=t}}requestAnimationFrame(tick)}}requestAnimationFrame(tick);
document.querySelectorAll('[data-tab]').forEach(b=>b.onclick=()=>{{document.querySelectorAll('[data-tab],.tab').forEach(x=>x.classList.remove('active'));b.classList.add('active');$('#'+b.dataset.tab).classList.add('active');resize()}});canvas.onwheel=e=>{{e.preventDefault();let before={{x:(e.offsetX-pan.x)/zoom,y:(e.offsetY-pan.y)/zoom}};zoom=Math.max(12,Math.min(120,zoom*(e.deltaY<0?1.12:.89)));pan={{x:e.offsetX-before.x*zoom,y:e.offsetY-before.y*zoom}};follow=false;draw()}};canvas.onpointerdown=e=>{{drag={{x:e.clientX,y:e.clientY,px:pan.x,py:pan.y,moved:false}};canvas.setPointerCapture(e.pointerId);canvas.classList.add('dragging')}};canvas.onpointermove=e=>{{if(!drag)return;let dx=e.clientX-drag.x,dy=e.clientY-drag.y;drag.moved||=Math.hypot(dx,dy)>4;pan={{x:drag.px+dx,y:drag.py+dy}};follow=false;draw()}};canvas.onpointerup=e=>{{let d=drag;drag=null;canvas.classList.remove('dragging');if(!d?.moved){{for(const [id,a] of Object.entries(current().agents))if(a.x!=null){{let p=xy(a.x,a.y);if(Math.hypot(e.offsetX-(p.x+zoom/2),e.offsetY-(p.y+zoom/2))<Math.max(12,zoom*.4)){{selected=id;follow=false;render();break}}}}}}}};addEventListener('keydown',e=>{{if(e.key==='ArrowLeft')set(i-1);if(e.key==='ArrowRight')set(i+1);if(e.key.toLowerCase()==='f'&&selected){{follow=!follow;render()}}}});resize();render();</script></body></html>'''
