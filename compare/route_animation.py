"""Playback of genuine evaluated-model routes in the shared DeliveryEnv.

The caller supplies a trace from ``compare.learning_view.replay``. This module
does not train a teaching model or simulate different rewards.
"""

import json


def animation_html(model, stage, trace):
    data = json.dumps({"model": model, "stage": stage, "trace": trace},
                      separators=(",", ":")).replace("</", "<\\/")
    return r'''<!doctype html><html><head><meta charset="utf-8"><style>
      *{box-sizing:border-box}body{font-family:Inter,system-ui,sans-serif;color:#172033;margin:0;background:#fff}
      .wrap{display:grid;grid-template-columns:minmax(220px,280px) minmax(200px,1fr);gap:16px;align-items:start}
      svg{display:block;width:100%;max-width:280px;height:auto;border:1px solid #dce3ed;border-radius:9px;background:#f8fafc}
      .controls{display:flex;gap:7px;flex-wrap:wrap;align-items:center;margin:10px 0}
      button{border:1px solid #cbd5e1;border-radius:8px;background:white;padding:7px 11px;color:#172033;cursor:pointer}
      button.primary{background:#2563eb;color:white;border-color:#2563eb}button:disabled{opacity:.45}
      .card{border:1px solid #e2e8f0;border-radius:10px;padding:12px;margin-bottom:9px;background:#fff}
      .card strong{font-size:16px}.muted{font-size:12px;color:#64748b}.good{color:#15803d}.bad{color:#b91c1c}
      .detail{margin-top:5px;font-size:13px}.bar{height:7px;border-radius:5px;background:#e2e8f0;overflow:hidden;margin-top:9px}
      .fill{height:100%;background:#2563eb}
      @media(max-width:560px){.wrap{grid-template-columns:1fr}svg{max-width:260px}}
    </style></head><body><div class="wrap"><div>
      <svg id="grid" viewBox="0 0 400 400" aria-label="Animated delivery route"></svg>
      <div class="controls"><button id="play" class="primary">▶ Play</button><button id="step">Step</button><button id="reset">Reset</button></div>
      <div class="muted">R = restaurant · C = customer · red = traffic jam · blue = rider</div>
    </div><div>
      <div class="card"><div class="muted">Algorithm and genuine checkpoint</div><strong id="model"></strong><div class="detail" id="stage"></div><div class="bar"><div id="progress" class="fill"></div></div></div>
      <div class="card"><div class="muted">Current move and feedback</div><strong id="action"></strong><div class="detail" id="feedback"></div><div class="detail" id="reward"></div><div class="detail" id="total"></div></div>
      <div class="card"><div class="muted">Delivery</div><strong id="result"></strong><div class="detail" id="steps"></div><div class="detail" id="minutes"></div></div>
    </div></div><script>
      const data=__DATA__, trace=data.trace, $=id=>document.getElementById(id), ns='http://www.w3.org/2000/svg';
      const svg=$('grid'), hot=new Set(trace.hotspots.map(p=>p.join(',')));
      let frame=0,timer=null;
      function el(tag,attrs){let x=document.createElementNS(ns,tag);for(let [k,v] of Object.entries(attrs))x.setAttribute(k,v);svg.appendChild(x);return x}
      for(let r=0;r<10;r++)for(let c=0;c<10;c++){let h=hot.has(r+','+c);el('rect',{x:c*40,y:r*40,width:40,height:40,fill:h?'#fee2e2':'#f8fafc',stroke:'#dce3ed'});if(h)el('text',{x:c*40+20,y:r*40+26,'text-anchor':'middle',fill:'#ef4444','font-size':17}).textContent='×'}
      function marker(p,color,label){el('circle',{cx:p[1]*40+20,cy:p[0]*40+20,r:15,fill:color});el('text',{x:p[1]*40+20,y:p[0]*40+25,'text-anchor':'middle',fill:'white','font-size':12,'font-weight':700}).textContent=label}
      marker(trace.start,'#15803d','R');marker(trace.dest,'#1d4ed8','C');
      const line=el('polyline',{points:'',fill:'none',stroke:'#2563eb','stroke-width':4,'stroke-linecap':'round','stroke-linejoin':'round'});
      const rider=el('circle',{cx:0,cy:0,r:9,fill:'#2563eb',stroke:'white','stroke-width':3});
      function stop(){if(timer){clearInterval(timer);timer=null}$('play').textContent='▶ Play'}
      function draw(){const p=trace.points[Math.min(frame,trace.points.length-1)], move=trace.moves[frame-1];
        line.setAttribute('points',trace.points.slice(0,frame+1).map(q=>(q[1]*40+20)+','+(q[0]*40+20)).join(' '));
        rider.setAttribute('cx',p[1]*40+20);rider.setAttribute('cy',p[0]*40+20);
        $('model').textContent=data.model;$('stage').textContent=data.stage.label+' · '+data.stage.episode+' completed episodes · seed '+data.stage.training_seed;
        $('progress').style.width=(100*frame/Math.max(1,trace.moves.length))+'%';
        $('action').textContent=move?('Move '+frame+': '+move.action):'Ready to start';
        $('feedback').textContent=move?move.feedback:'Use Play or Step to follow the evaluated route.';
        $('reward').textContent='Current reward: '+(move?move.reward:'—');
        $('total').textContent='Cumulative reward: '+(move?move.total:'0.00');
        $('result').textContent=frame===trace.moves.length?(trace.delivered?'Delivered':'Not delivered'):'Route in progress';
        $('result').className=frame===trace.moves.length?(trace.delivered?'good':'bad'):'';
        $('steps').textContent='Steps: '+frame+' / '+trace.steps;
        $('minutes').textContent='Simulated time: '+(move?move.minutes:'0.0')+' min';
        $('step').disabled=frame>=trace.moves.length;
      }
      $('step').onclick=()=>{stop();if(frame<trace.moves.length){frame++;draw()}};
      $('reset').onclick=()=>{stop();frame=0;draw()};
      $('play').onclick=()=>{if(timer){stop();return}if(frame>=trace.moves.length)frame=0;draw();$('play').textContent='❚❚ Pause';timer=setInterval(()=>{frame++;draw();if(frame>=trace.moves.length)stop()},420)};
      draw();</script></body></html>'''.replace("__DATA__", data)
