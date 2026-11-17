"""Small, isolated DQN training demonstration for the dashboard's 10x10 grid.

This trains a fresh model on one fixed toy delivery map. It does not update any
checkpoint used for the project's evaluated routing models.
"""

from collections import deque
import json
import random

import numpy as np
import torch
from torch import nn

from dqn.dqn_v4 import QNetwork


MOVES = ((-1, 0), (1, 0), (0, -1), (0, 1))
ACTION_NAMES = ("Up", "Down", "Left", "Right")
MAX_MOVES = 40
MILESTONES = (0, 60, 150, 300)


class DemoGrid:
    def __init__(self, start, dest, hotspots):
        self.start = tuple(start)
        self.dest = tuple(dest)
        self.hotspots = np.asarray(hotspots, dtype=bool)
        if self.hotspots.shape != (10, 10):
            raise ValueError("The learning demo expects a 10x10 grid")
        self.reset()

    def reset(self):
        self.pos = self.start
        self.steps = 0
        return self.state()

    def state(self):
        r, c = self.pos
        dr, dc = self.dest
        near = []
        for vr, vc in MOVES:
            nr, nc = r + vr, c + vc
            near.append(float(0 <= nr < 10 and 0 <= nc < 10 and self.hotspots[nr, nc]))
        return np.array((r / 9, c / 9, dr / 9, dc / 9,
                         (dr - r) / 9, (dc - c) / 9,
                         self.steps / MAX_MOVES, *near), dtype=np.float32)

    def step(self, action):
        old_dist = abs(self.pos[0] - self.dest[0]) + abs(self.pos[1] - self.dest[1])
        vr, vc = MOVES[action]
        nr, nc = self.pos[0] + vr, self.pos[1] + vc
        wall = not (0 <= nr < 10 and 0 <= nc < 10)
        if not wall:
            self.pos = (nr, nc)
        self.steps += 1
        jam = bool(self.hotspots[self.pos])
        new_dist = abs(self.pos[0] - self.dest[0]) + abs(self.pos[1] - self.dest[1])
        # Step cost, congestion cost, and progress feedback; arrival is rewarded.
        reward = -1.0 - 2.0 * jam - 2.0 * wall + 0.8 * (old_dist - new_dist)
        delivered = self.pos == self.dest
        timeout = self.steps >= MAX_MOVES and not delivered
        if delivered:
            reward += 35.0
        elif timeout:
            reward -= 10.0
        feedback = "Delivered" if delivered else ("Time limit" if timeout else
                   "Map edge" if wall else "Traffic jam" if jam else
                   "Closer to customer" if new_dist < old_dist else "Moved away")
        return self.state(), reward, delivered or timeout, delivered, feedback


def _trace(net, grid):
    state = grid.reset()
    points = [list(grid.pos)]
    moves = []
    total = 0.0
    delivered = False
    for _ in range(MAX_MOVES):
        with torch.no_grad():
            action = int(net(torch.from_numpy(state)).argmax().item())
        state, reward, done, delivered, feedback = grid.step(action)
        total += reward
        moves.append({"action": ACTION_NAMES[action], "reward": round(reward, 2),
                      "feedback": feedback, "total": round(total, 2)})
        points.append(list(grid.pos))
        if done:
            break
    return {"points": points, "moves": moves, "delivered": delivered,
            "total_reward": round(total, 2), "steps": len(moves)}


def train_demo(start, dest, hotspot_coords, episodes=300, seed=17):
    """Train standard DQN with replay and target network; return replayable traces."""
    if episodes < 1:
        raise ValueError("episodes must be positive")
    hot = np.zeros((10, 10), dtype=bool)
    for r, c in hotspot_coords:
        hot[int(r), int(c)] = True
    grid = DemoGrid(start, dest, hot)
    torch.manual_seed(seed)
    rng = random.Random(seed)
    online = QNetwork(obs_dim=11, hidden=64)
    target = QNetwork(obs_dim=11, hidden=64)
    target.load_state_dict(online.state_dict())
    optimizer = torch.optim.Adam(online.parameters(), lr=0.002)
    replay = deque(maxlen=5000)
    history = []
    snapshots = []
    updates = 0

    def save_snapshot(ep):
        trace = _trace(online, grid)
        recent = history[-30:]
        snapshots.append({"episode": ep, "recent_success":
                          round(100 * sum(row["delivered"] for row in recent) / len(recent))
                          if recent else None,
                          "recent_reward": round(sum(row["reward"] for row in recent) / len(recent), 1)
                          if recent else None, **trace})

    save_snapshot(0)
    for ep in range(1, episodes + 1):
        state = grid.reset()
        ep_reward = 0.0
        eps = max(0.05, 1.0 - 0.95 * (ep - 1) / max(1, episodes * 0.8))
        for _ in range(MAX_MOVES):
            if rng.random() < eps:
                action = rng.randrange(4)
            else:
                with torch.no_grad():
                    action = int(online(torch.from_numpy(state)).argmax().item())
            next_state, reward, done, delivered, _ = grid.step(action)
            replay.append((state, action, reward / 10.0, next_state, float(done)))
            ep_reward += reward
            state = next_state
            if len(replay) >= 64 and grid.steps % 4 == 0:
                batch = rng.sample(replay, 64)
                s, a, r, s2, terminal = zip(*batch)
                states = torch.from_numpy(np.stack(s))
                next_states = torch.from_numpy(np.stack(s2))
                selected_q = online(states).gather(1, torch.tensor(a).unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    target_q = target(next_states).max(dim=1).values
                    goal_q = torch.tensor(r, dtype=torch.float32) + 0.97 * target_q * (1.0 - torch.tensor(terminal))
                loss = nn.functional.smooth_l1_loss(selected_q, goal_q)
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(online.parameters(), 10.0)
                optimizer.step()
                updates += 1
                if updates % 100 == 0:
                    target.load_state_dict(online.state_dict())
            if done:
                break
        history.append({"episode": ep, "reward": round(ep_reward, 2), "delivered": delivered})
        if ep in MILESTONES or ep == episodes:
            save_snapshot(ep)
    return {"start": list(start), "dest": list(dest),
            "hotspots": [list(p) for p in hotspot_coords],
            "snapshots": snapshots, "history": history, "updates": updates}


def animation_html(result):
    """Return a self-contained route player with real per-move training feedback."""
    data = json.dumps(result, separators=(",", ":"))
    return r'''<!doctype html><html><head><meta charset="utf-8"><style>
      *{box-sizing:border-box}body{font-family:Inter,system-ui,sans-serif;color:#172033;margin:0;background:#fff}
      .wrap{display:grid;grid-template-columns:minmax(300px,430px) minmax(210px,1fr);gap:20px;align-items:start}
      svg{display:block;width:100%;max-width:430px;height:auto;border:1px solid #dce3ed;border-radius:10px;background:#f8fafc}
      .controls{display:flex;gap:8px;flex-wrap:wrap;align-items:center;margin:12px 0}.controls button,.controls select{border:1px solid #cbd5e1;border-radius:8px;background:white;padding:8px 11px;color:#172033;cursor:pointer}
      .controls button.primary{background:#2563eb;color:#fff;border-color:#2563eb}.controls button:disabled{opacity:.45;cursor:default}
      .card{border:1px solid #e2e8f0;border-radius:10px;padding:12px;margin-bottom:10px;background:#fff}.card strong{font-size:18px}
      .muted{font-size:12px;color:#64748b}.good{color:#15803d}.bad{color:#b91c1c}
      .bar{height:7px;border-radius:5px;background:#e2e8f0;overflow:hidden;margin:8px 0}.fill{height:100%;background:#2563eb}
      @media(max-width:640px){.wrap{grid-template-columns:1fr}}
    </style></head><body><div class="wrap"><div><svg id="grid" viewBox="0 0 400 400" aria-label="Animated DQN training route"></svg>
      <div class="controls"><label>Training stage <select id="stage"></select></label><button id="play" class="primary">▶ Play</button><button id="step">Step</button><button id="reset">Reset</button></div>
      <div class="muted">R = restaurant · C = customer · pink = traffic jam · blue = rider</div></div>
      <div><div class="card"><div class="muted">Training progress</div><strong id="stageLabel"></strong><div id="recent"></div><div class="bar"><div id="progress" class="fill"></div></div><div class="muted">The DQN learns from rewards stored in replay memory. Routes at each stage are greedy evaluations of that checkpoint.</div></div>
      <div class="card"><div class="muted">Current move</div><strong id="move"></strong><div id="feedback"></div><div id="stepReward"></div><div id="total"></div></div>
      <div class="card"><div class="muted">Route result</div><strong id="result"></strong><div id="steps"></div><div class="muted">A single sample route can fluctuate even as average training improves. This is a fixed-grid teaching demo, separate from the evaluated DQN model above.</div></div></div></div>
      <script>const data=__DATA__;
      const $=id=>document.getElementById(id), ns='http://www.w3.org/2000/svg';
      const svg=$('grid'), stage=$('stage'); let index=0, frame=0, timer=null;
      const hot=new Set(data.hotspots.map(p=>p.join(',')));
      function el(tag,attrs){let x=document.createElementNS(ns,tag);for(let [k,v] of Object.entries(attrs))x.setAttribute(k,v);svg.appendChild(x);return x}
      for(let r=0;r<10;r++)for(let c=0;c<10;c++){let h=hot.has(r+','+c);el('rect',{x:c*40,y:r*40,width:40,height:40,fill:h?'#fee2e2':'#f8fafc',stroke:'#dce3ed'});if(h)el('text',{x:c*40+20,y:r*40+25,'text-anchor':'middle',fill:'#ef4444','font-size':16}).textContent='×'}
      function marker(p,color,label){el('circle',{cx:p[1]*40+20,cy:p[0]*40+20,r:14,fill:color});el('text',{x:p[1]*40+20,y:p[0]*40+25,'text-anchor':'middle',fill:'white','font-size':12,'font-weight':700}).textContent=label}
      marker(data.start,'#15803d','R');marker(data.dest,'#1d4ed8','C');
      let line=el('polyline',{points:'',fill:'none',stroke:'#2563eb','stroke-width':4,'stroke-linecap':'round','stroke-linejoin':'round'});
      let rider=el('circle',{cx:0,cy:0,r:9,fill:'#2563eb',stroke:'white','stroke-width':3});
      function stop(){if(timer){clearInterval(timer);timer=null}$('play').textContent='▶ Play'}
      function draw(){let snap=data.snapshots[index], p=snap.points[Math.min(frame,snap.points.length-1)], move=snap.moves[frame-1];
        line.setAttribute('points',snap.points.slice(0,frame+1).map(q=>(q[1]*40+20)+','+(q[0]*40+20)).join(' '));
        rider.setAttribute('cx',p[1]*40+20);rider.setAttribute('cy',p[0]*40+20);
        $('stageLabel').textContent='After '+snap.episode+' training episodes';
        $('recent').textContent=snap.recent_success===null?'Untrained network':('Last 30 training trips: '+snap.recent_success+'% delivered · average reward '+snap.recent_reward);
        $('progress').style.width=(100*frame/Math.max(1,snap.moves.length))+'%';
        $('move').textContent=move?('Move '+frame+': '+move.action):'Ready to start';
        $('feedback').textContent=move?move.feedback:'Watch the rider move, then see the feedback.';
        $('stepReward').textContent=move?('Reward for move: '+move.reward):'';
        $('total').textContent='Reward so far: '+(move?move.total:'0.00');
        $('result').textContent=frame===snap.moves.length?(snap.delivered?'Delivered':'Not delivered'):'Route in progress';
        $('result').className=frame===snap.moves.length?(snap.delivered?'good':'bad'):'';
        $('steps').textContent=frame===snap.moves.length?(snap.steps+' moves · total reward '+snap.total_reward):('Move '+frame+' of '+snap.steps);
        $('step').disabled=frame>=snap.moves.length;
      }
      data.snapshots.forEach((s,i)=>{let o=document.createElement('option');o.value=i;o.textContent=s.episode===0?'Before training':'After '+s.episode+' episodes';stage.appendChild(o)});
      stage.onchange=()=>{stop();index=Number(stage.value);frame=0;draw()};
      $('step').onclick=()=>{stop();if(frame<data.snapshots[index].moves.length){frame++;draw()}};
      $('reset').onclick=()=>{stop();frame=0;draw()};
      $('play').onclick=()=>{if(timer){stop();return}if(frame>=data.snapshots[index].moves.length)frame=0;draw();$('play').textContent='❚❚ Pause';timer=setInterval(()=>{frame++;draw();if(frame>=data.snapshots[index].moves.length)stop()},420)};
      draw();</script></body></html>'''.replace("__DATA__", data)
