#!/usr/bin/env python3
# experiment.py
"""
Comparativa rápida entre agente clásico tabular y agente basado en QNN.
Modo: igualar por tiempo (wall-clock) o por episodios (número de partidas).
Genera CSV con métricas por episodio y un resumen impreso.

Requisitos:
- numpy, matplotlib, pandas, torch, qiskit (si vas a usar el agente cuántico)
- Ajusta parámetros de QuantumPongAgent si simular es muy lento (num_qubits, reps, backend).
"""

import time
import argparse
import csv
import os
import statistics
import numpy as np
from time import sleep
from random import randint
from math import ceil, floor
import matplotlib.pyplot as plt
import pandas as pd

# Intento de import qiskit/torch; el script funcionará con solo el agente clásico si faltan libs.
try:
    import torch, torch.nn as nn, torch.optim as optim
    from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
    from qiskit_machine_learning.neural_networks import EstimatorQNN
    from qiskit_machine_learning.connectors import TorchConnector
    from qiskit import Aer
    QISKIT_AVAILABLE = True
except Exception as e:
    QISKIT_AVAILABLE = False


# ---------------------------
# Entorno
# ---------------------------
class PongEnvironment:
    
    def __init__(self, max_life=3, height_px = 40, width_px = 50, movimiento_px = 3):
        
        self.action_space = ['Arriba','Abajo']
        
        self._step_penalization = 0
        
        self.state = [0,0,0]
        
        self.total_reward = 0
        
        self.dx = movimiento_px
        self.dy = movimiento_px
        
        filas = ceil(height_px/movimiento_px)
        columnas = ceil(width_px/movimiento_px)
        
        self.positions_space = np.array([[[0 for z in range(columnas)] for y in range(filas)] for x in range(filas)])

        self.lives = max_life
        self.max_life=max_life
        
        self.x = randint(int(width_px/2), width_px) 
        self.y = randint(0, height_px-10)
        
        self.player_alto = int(height_px/4)

        self.player1 = self.player_alto  # posic. inicial del player
        
        self.score = 0
        
        self.width_px = width_px
        self.height_px = height_px
        self.radio = 2.5

    def reset(self):
        self.total_reward = 0
        self.state = [0,0,0]
        self.lives = self.max_life
        self.score = 0
        self.x = randint(int(self.width_px/2), self.width_px) 
        self.y = randint(0, self.height_px-10)
        return self.state

    def step(self, action, animate=False):
        self._apply_action(action, animate)
        done = self.lives <=0 # final
        reward = self.score
        reward += self._step_penalization
        self.total_reward += reward
        return self.state, reward , done

    def _apply_action(self, action, animate=False):
        
        if action == "Arriba":
            self.player1 += abs(self.dy)
        elif action == "Abajo":
            self.player1 -= abs(self.dy)
            
        self.avanza_player()

        self.avanza_frame()

        if animate:
            if not hasattr(self, '_fig_ax'):
                self._fig_ax = self.dibujar_frame()  # crea la figura y eje la primera vez
            else:
                self._fig_ax = self.dibujar_frame(self._fig_ax)  # refresca la misma figura

        self.state = (floor(self.player1/abs(self.dy))-2, floor(self.y/abs(self.dy))-2, floor(self.x/abs(self.dx))-2)
    
    def detectaColision(self, ball_y, player_y):
        if (player_y+self.player_alto >= (ball_y-self.radio)) and (player_y <= (ball_y+self.radio)):
            return True
        else:
            return False
    
    def avanza_player(self):
        if self.player1 + self.player_alto >= self.height_px:
            self.player1 = self.height_px - self.player_alto
        elif self.player1 <= -abs(self.dy):
            self.player1 = -abs(self.dy)

    def avanza_frame(self):
        self.x += self.dx
        self.y += self.dy
        if self.x <= 3 or self.x > self.width_px:
            self.dx = -self.dx
            if self.x <= 3:
                ret = self.detectaColision(self.y, self.player1)

                if ret:
                    self.score = 10
                else:
                    self.score = -10
                    self.lives -= 1
                    if self.lives>0:
                        self.x = randint(int(self.width_px/2), self.width_px)
                        self.y = randint(0, self.height_px-10)
                        self.dx = abs(self.dx)
                        self.dy = abs(self.dy)
        else:
            self.score = 0

        if self.y < 0 or self.y > self.height_px:
            self.dy = -self.dy

    def dibujar_frame(self, fig_ax=None):
        # Si no existe la figura, crearla
        if fig_ax is None:
            fig, ax = plt.subplots(figsize=(5,4))
        else:
            fig, ax = fig_ax
            ax.clear()  # limpiar el frame anterior

        # dibujar la bola
        circle = plt.Circle((self.x, self.y), self.radio, fc='slategray', ec="black")
        ax.add_patch(circle)
        # dibujar el jugador
        rectangle = plt.Rectangle((-5, self.player1), 5, self.player_alto, fc='gold', ec="none")
        ax.add_patch(rectangle)

        # límites y texto
        ax.set_xlim(-5, self.width_px+5)
        ax.set_ylim(-5, self.height_px+5)
        ax.text(4, self.height_px, f"SCORE:{self.total_reward}  LIFE:{self.lives}", fontsize=12)
        if self.lives <=0:
            ax.text(10, self.height_px-14, "GAME OVER", fontsize=16)
        elif self.total_reward >= 1000:
            ax.text(10, self.height_px-14, "YOU WIN!", fontsize=16)

        plt.pause(0.001)  # pausa corta para que se refresque la ventana
        sleep(0.02)
        return fig, ax

# ---------------------------
# Agente clásico
# ---------------------------
class PongAgent:
    def __init__(self, game, policy=None, discount_factor = 0.1, learning_rate = 0.1, ratio_explotacion = 0.9):
        if policy is not None:
            self._q_table = policy
        else:
            position = list(game.positions_space.shape)
            position.append(len(game.action_space))
            self._q_table = np.zeros(position)
        self.discount_factor = discount_factor
        self.learning_rate = learning_rate
        self.ratio_explotacion = ratio_explotacion
        # contadores para la comparativa
        self.forward_calls = 0
        self.backward_calls = 0

    def get_next_step(self, state, game):
        self.forward_calls += 1
        next_step = np.random.choice(list(game.action_space))
        if np.random.uniform() <= self.ratio_explotacion:
            idx_action = np.random.choice(np.flatnonzero(
                    self._q_table[state[0],state[1],state[2]] == self._q_table[state[0],state[1],state[2]].max()
                ))
            next_step = list(game.action_space)[idx_action]
        return next_step

    def update(self, game, old_state, action_taken, reward_action_taken, new_state, reached_end):
        self.backward_calls += 1
        idx_action_taken =list(game.action_space).index(action_taken)
        actual_q_value_options = self._q_table[old_state[0], old_state[1], old_state[2]]
        actual_q_value = actual_q_value_options[idx_action_taken]
        future_q_value_options = self._q_table[new_state[0], new_state[1], new_state[2]]
        future_max_q_value = reward_action_taken  +  self.discount_factor*future_q_value_options.max()
        if reached_end:
            future_max_q_value = reward_action_taken
        self._q_table[old_state[0], old_state[1], old_state[2], idx_action_taken] = actual_q_value + \
                                              self.learning_rate*(future_max_q_value -actual_q_value)

    def get_policy(self):
        return self._q_table

# ---------------------------
# Agente cuántico
# ---------------------------
class QuantumPongAgent:
    def __init__(self, game, discount_factor=0.1, learning_rate=0.01, ratio_explotacion=0.9,
                 num_qubits=3, reps=1):
        if not QISKIT_AVAILABLE:
            raise RuntimeError("Qiskit/Torch no están disponibles en este entorno.")
        self.discount_factor = discount_factor
        self.learning_rate = learning_rate
        self.ratio_explotacion = ratio_explotacion
        self.action_space = game.action_space
        self.num_actions = len(game.action_space)
        self.num_qubits = num_qubits
        # instrumentation
        self.forward_calls = 0
        self.backward_calls = 0
        # construir circuito simple (ajusta num_qubits/reps para tiempo)
        feature_map = ZZFeatureMap(self.num_qubits)
        ansatz = RealAmplitudes(self.num_qubits, reps=reps)
        qc = feature_map.compose(ansatz)
        # QNN por acción (simple y directo)
        self.qnns = [EstimatorQNN(circuit=qc,
                                  input_params=feature_map.parameters,
                                  weight_params=ansatz.parameters) 
                     for _ in range(self.num_actions)]
        self.models = [TorchConnector(qnn) for qnn in self.qnns]
        self.optimizers = [optim.Adam(model.parameters(), lr=learning_rate) for model in self.models]
        self.loss_fn = nn.MSELoss()

    def state_to_tensor(self, state):
        # escalado sencillo; ajústalo si tu feature map espera otro rango
        return torch.tensor([s / 50 for s in state], dtype=torch.float32)

    def get_next_step(self, state):
        state_tensor = self.state_to_tensor(state)
        q_vals = []
        for model in self.models:
            self.forward_calls += 1
            out = model(state_tensor)
            # convertir a float escalar
            val = out.detach().cpu().numpy()
            q_vals.append(float(np.asarray(val).reshape(-1)[0]))
        q_tensor = np.array(q_vals)
        if np.random.uniform() <= self.ratio_explotacion:
            return self.action_space[int(np.argmax(q_tensor))]
        else:
            return np.random.choice(self.action_space)

    def update(self, state, action_taken, reward, next_state, done):
        state_tensor = self.state_to_tensor(state)
        next_state_tensor = self.state_to_tensor(next_state)
        q_vals = []
        q_next = []
        for model in self.models:
            self.forward_calls += 1
            q_vals.append(float(np.asarray(model(state_tensor).detach().cpu().numpy()).reshape(-1)[0]))
            self.forward_calls += 1
            q_next.append(float(np.asarray(model(next_state_tensor).detach().cpu().numpy()).reshape(-1)[0]))

        target = reward
        if not done:
            target += self.discount_factor * max(q_next)

        action_index = self.action_space.index(action_taken)
        # backprop solo para la QNN de esa acción
        self.optimizers[action_index].zero_grad()
        self.backward_calls += 1
        output = self.models[action_index](state_tensor)
        loss = self.loss_fn(output, torch.tensor([target], dtype=torch.float32))
        loss.backward()
        self.optimizers[action_index].step()

# ---------------------------
# Runner de experimentos
# ---------------------------
def run_single(agent_type, mode, budget, seed, agent_kwargs, game_kwargs):
    """
    agent_type: 'classical' or 'quantum'
    mode: 'time' or 'episodes'
    budget: seconds if mode=='time', episodes if mode=='episodes'
    seed: int
    """
    np.random.seed(seed)
    # crear entorno y agente
    game = PongEnvironment(**game_kwargs)
    if agent_type == 'classical':
        agent = PongAgent(game, **agent_kwargs)
    elif agent_type == 'quantum':
        if not QISKIT_AVAILABLE:
            raise RuntimeError("Qiskit/Torch no disponibles para agente cuántico.")
        agent = QuantumPongAgent(game, **agent_kwargs)
    else:
        raise ValueError("agent_type must be 'classical' or 'quantum'")

    # métricas por episodio
    ep_rewards = []
    ep_times = []
    ep_forward_calls = []
    ep_backward_calls = []

    start_total = time.time()
    episodes_done = 0

    # loop principal: finaliza por el modo elegido
    while True:
        # stop condition
        if mode == 'time' and (time.time() - start_total) >= budget:
            break
        if mode == 'episodes' and episodes_done >= budget:
            break

        s = game.reset()
        done = False
        ep_start = time.time()
        steps = 0
        # jugar un episodio
        while not done and steps < 3000 and game.total_reward <= 1000:
            if agent_type == 'classical':
                a = agent.get_next_step(s, game)
            else:
                a = agent.get_next_step(s)
            next_s, r, done = game.step(a)
            # actualizar (classical y quantum tienen signatura ligeramente distinta)
            if agent_type == 'classical':
                agent.update(game, s, a, r, next_s, done)
            else:
                agent.update(s, a, r, next_s, done)
            s = next_s
            steps += 1
        ep_time = time.time() - ep_start
        ep_rewards.append(game.total_reward)
        ep_times.append(ep_time)
        ep_forward_calls.append(getattr(agent, 'forward_calls', 0))
        ep_backward_calls.append(getattr(agent, 'backward_calls', 0))
        episodes_done += 1

    total_time = time.time() - start_total
    # resumen
    summary = {
        'agent_type': agent_type,
        'seed': seed,
        'episodes_done': episodes_done,
        'total_time_s': total_time,
        'avg_reward_per_ep': float(np.mean(ep_rewards)) if ep_rewards else 0.0,
        'std_reward_per_ep': float(np.std(ep_rewards)) if ep_rewards else 0.0,
        'avg_time_per_ep': float(np.mean(ep_times)) if ep_times else 0.0,
        'total_forward_calls': int(ep_forward_calls[-1]) if ep_forward_calls else 0,
        'total_backward_calls': int(ep_backward_calls[-1]) if ep_backward_calls else 0,
        'rewards': ep_rewards,
        'times': ep_times
    }
    return summary

def aggregate_and_print(results, name):
    seeds = [r['seed'] for r in results]
    episodes = [r['episodes_done'] for r in results]
    times = [r['total_time_s'] for r in results]
    avg_rewards = [r['avg_reward_per_ep'] for r in results]
    print(f"\n=== RESUMEN {name} ===")
    print(f"Repeticiones: {len(results)}, Seeds: {seeds}")
    print(f"Episodes per run: mean={statistics.mean(episodes):.1f} std={statistics.pstdev(episodes):.1f}")
    print(f"Tiempo total (s): mean={statistics.mean(times):.1f} std={statistics.pstdev(times):.1f}")
    print(f"Avg reward/ep: mean={statistics.mean(avg_rewards):.3f} std={statistics.pstdev(avg_rewards):.3f}")
    # devolver tabla pandas
    return pd.DataFrame([{
        'seed': r['seed'],
        'episodes_done': r['episodes_done'],
        'total_time_s': r['total_time_s'],
        'avg_reward_per_ep': r['avg_reward_per_ep'],
        'std_reward_per_ep': r['std_reward_per_ep'],
        'avg_time_per_ep': r['avg_time_per_ep'],
        'total_forward_calls': r['total_forward_calls'],
        'total_backward_calls': r['total_backward_calls']
    } for r in results])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['time','episodes'], default='time', help='Modo de igualación')
    parser.add_argument('--budget', type=float, default=600.0, help='Segundos (time) o episodios (episodes)')
    parser.add_argument('--repeats', type=int, default=3, help='Repeticiones por agente (seeds)')
    parser.add_argument('--out', default='results', help='Directorio de salida para CSV/plots')
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    seeds = list(range(args.repeats))
    classical_results = []
    quantum_results = []

    # parámetros (ajusta aquí si QNN simula muy lento)
    game_kwargs = {'max_life': 3, 'height_px': 40, 'width_px': 50, 'movimiento_px': 3}
    classical_kwargs = {'discount_factor':0.2, 'learning_rate':0.1, 'ratio_explotacion':0.85}
    quantum_kwargs = {'discount_factor':0.2, 'learning_rate':0.01, 'ratio_explotacion':0.85, 'num_qubits':3, 'reps':1}

    for s in seeds:
        print(f"\nRunning classical, seed={s}")
        res_c = run_single('classical', args.mode, args.budget, s, classical_kwargs, game_kwargs)
        classical_results.append(res_c)
        print(f"  -> done episodes={res_c['episodes_done']} time={res_c['total_time_s']:.1f}s avg_r={res_c['avg_reward_per_ep']:.2f}")

        if QISKIT_AVAILABLE:
            print(f"Running quantum, seed={s}")
            res_q = run_single('quantum', args.mode, args.budget, s, quantum_kwargs, game_kwargs)
            quantum_results.append(res_q)
            print(f"  -> done episodes={res_q['episodes_done']} time={res_q['total_time_s']:.1f}s avg_r={res_q['avg_reward_per_ep']:.2f} fwd={res_q['total_forward_calls']}")
        else:
            print("Skipping quantum run: Qiskit/Torch no disponibles.")
            quantum_results = []

    # resumen e impresión
    df_classic = aggregate_and_print(classical_results, "CLÁSICO")
    df_classic.to_csv(os.path.join(args.out, "classical_summary.csv"), index=False)
    if quantum_results:
        df_quantum = aggregate_and_print(quantum_results, "CUÁNTICO")
        df_quantum.to_csv(os.path.join(args.out, "quantum_summary.csv"), index=False)

    # guardar detalle de rewards por episodio por run
    def save_detail(results, fname):
        rows = []
        for r in results:
            for i, rew in enumerate(r['rewards']):
                rows.append({'seed': r['seed'], 'ep': i, 'reward': rew})
        pd.DataFrame(rows).to_csv(fname, index=False)

    save_detail(classical_results, os.path.join(args.out, "classical_rewards.csv"))
    if quantum_results:
        save_detail(quantum_results, os.path.join(args.out, "quantum_rewards.csv"))

    # plot ejemplo (reward medio por episodio across seeds)
    try:
        fig, ax = plt.subplots()
        # clásico
        dfc = pd.read_csv(os.path.join(args.out, "classical_rewards.csv"))
        gp = dfc.groupby('ep')['reward'].agg(['mean','std']).reset_index()
        ax.plot(gp['ep'], gp['mean'], label='clásico')
        # cuántico
        if quantum_results:
            dfq = pd.read_csv(os.path.join(args.out, "quantum_rewards.csv"))
            gq = dfq.groupby('ep')['reward'].agg(['mean','std']).reset_index()
            ax.plot(gq['ep'], gq['mean'], label='cuántico')
        ax.set_xlabel('episodio')
        ax.set_ylabel('reward')
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(args.out, "learning_curves.png"))
        print(f"Gráfica guardada en {args.out}/learning_curves.png")
    except Exception as e:
        print("No se pudo generar la gráfica:", e)

if __name__ == '__main__':
    main()
