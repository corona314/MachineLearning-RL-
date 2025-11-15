import numpy as np
import matplotlib.pyplot as plt
from random import randint
from time import sleep
from IPython.display import clear_output
from math import ceil,floor
from qiskit import Aer
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.connectors import TorchConnector
import torch
import torch.nn as nn
import torch.optim as optim

class QuantumPongAgent:
    def __init__(self, game, discount_factor=0.1, learning_rate=0.01, ratio_explotacion=0.9):
        self.discount_factor = discount_factor
        self.learning_rate = learning_rate
        self.ratio_explotacion = ratio_explotacion
        self.action_space = game.action_space
        self.num_actions = len(game.action_space)
        self.num_qubits = 3
        
        feature_map = ZZFeatureMap(self.num_qubits)
        ansatz = RealAmplitudes(self.num_qubits, reps=1)
        qc = feature_map.compose(ansatz)

        # --- QNNs independientes por acción ---
        self.qnns = [EstimatorQNN(circuit=qc,
                                  input_params=feature_map.parameters,
                                  weight_params=ansatz.parameters) 
                     for _ in range(self.num_actions)]
        self.models = [TorchConnector(qnn) for qnn in self.qnns]
        self.optimizers = [optim.Adam(model.parameters(), lr=learning_rate) for model in self.models]
        self.loss_fn = nn.MSELoss()

    def state_to_tensor(self, state):
        return torch.tensor([s / 50 for s in state], dtype=torch.float32)

    def get_next_step(self, state):
        state_tensor = self.state_to_tensor(state)
        q_values = torch.tensor([model(state_tensor).item() for model in self.models])
        if np.random.uniform() <= self.ratio_explotacion:
            return self.action_space[torch.argmax(q_values).item()]
        else:
            return np.random.choice(self.action_space)

    def update(self, state, action_taken, reward, next_state, done):
        state_tensor = self.state_to_tensor(state)
        next_state_tensor = self.state_to_tensor(next_state)

        q_values = torch.tensor([model(state_tensor).item() for model in self.models])
        q_next = torch.tensor([model(next_state_tensor).item() for model in self.models])

        target = reward
        if not done:
            target += self.discount_factor * torch.max(q_next)

        action_index = self.action_space.index(action_taken)
        # --- Backprop solo para la QNN de esa acción ---
        self.optimizers[action_index].zero_grad()
        output = self.models[action_index](state_tensor)
        loss = self.loss_fn(output, torch.tensor([target], dtype=torch.float32))
        loss.backward()
        self.optimizers[action_index].step()

    def get_q_table(self, game):
        """
        Construye una 'Q-table' aproximada evaluando cada QNN por acción
        sobre todos los estados discretos definidos en game.positions_space.
        Devuelve un ndarray de forma (filas, filas, columnas, num_actions).
        """
        pos_shape = game.positions_space.shape  # (filas, filas, columnas)
        q_table = np.zeros((pos_shape[0], pos_shape[1], pos_shape[2], self.num_actions), dtype=float)

        # Para reconstruir valores de estado en píxeles: en tu env usas floor(val/dy)-2,
        # por tanto invertimos: val ≈ (idx + 2) * abs(dy)
        step_x = abs(game.dx)
        step_y = abs(game.dy)

        for i in range(pos_shape[0]):        # player index
            for j in range(pos_shape[1]):    # ball y index
                for k in range(pos_shape[2]):# ball x index
                    # reconstruimos estado aproximado en píxeles
                    player_px = (i + 2) * step_y
                    ball_y_px  = (j + 2) * step_y
                    ball_x_px  = (k + 2) * step_x
                    state = (player_px, ball_y_px, ball_x_px)

                    # convertir y evaluar cada modelo
                    state_tensor = self.state_to_tensor(state)
                    # TorchConnector puede devolver tensores 0-d o de tamaño 1; convertimos a float
                    q_vals = []
                    for model in self.models:
                        out = model(state_tensor)
                        # out puede ser tensor shape [1] o [1,1]; extraemos escalar float
                        q_val = out.detach().cpu().numpy()
                        q_val = float(np.asarray(q_val).reshape(-1)[0])
                        q_vals.append(q_val)

                    q_table[i, j, k, :] = np.array(q_vals, dtype=float)

        return q_table

    def print_q_table(self, game, digits=2, show_best_action=True):
        """
        Imprime la Q-table redondeada para cada estado discreto.
        Si show_best_action=True, imprime también la acción elegida por la QNN.
        """
        q_table = self.get_q_table(game)
        filas, filas2, columnas, _ = q_table.shape
        # comprobación rápida
        assert filas == filas2, "Esperaba posiciones cuadradas (filas, filas, columnas)."

        np.set_printoptions(precision=digits, suppress=True)

        for i in range(filas):
            print(f"Player idx = {i}")
            for j in range(filas):
                row_str = ""
                for k in range(columnas):
                    q_vals = q_table[i, j, k]
                    best_idx = int(np.argmax(q_vals))
                    q_str = "[" + ", ".join(f"{v:.{digits}f}" for v in q_vals) + "]"
                    if show_best_action:
                        row_str += f"{q_str}->{self.action_space[best_idx]}  "
                    else:
                        row_str += f"{q_str}  "
                print(row_str)
            print("-" * 80)


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
 


def play_quantum(rounds=50, max_life=3, discount_factor = 0.1, learning_rate = 0.01,
                 ratio_explotacion=0.9, learner=None, game=None, animate=False):

    if game is None:
        game = PongEnvironment(max_life=max_life, movimiento_px = 3)
        
    if learner is None:
        print("Begin new Quantum Train!")
        learner = QuantumPongAgent(game, discount_factor=discount_factor, learning_rate=learning_rate, ratio_explotacion=ratio_explotacion)

    max_points = -9999
    first_max_reached = 0
    total_rw = 0
    steps = []

    for played_games in range(rounds):
        state = game.reset()
        done = False
        itera = 0

        while not done and itera < 3000 and game.total_reward <= 1000:
            action = learner.get_next_step(state)
            next_state, reward, done = game.step(action, animate=animate)
            learner.update(state, action, reward, next_state, done)
            state = next_state
            itera += 1

        steps.append(itera)
        total_rw += game.total_reward

        if game.total_reward > max_points:
            max_points = game.total_reward
            first_max_reached = played_games

        if played_games % 10 == 0 and played_games > 0 and not animate:
            print(f"-- Partidas[{played_games}] Avg.Puntos[{int(total_rw/(played_games+1))}] "
                  f"AVG Steps[{int(np.array(steps).mean())}] Max Score[{max_points}]")

    print(f'Partidas[{played_games}] Avg.Puntos[{int(total_rw/(played_games+1))}] '
          f'Max score[{max_points}] en partida[{first_max_reached}]')

    return learner, game


demo_game = PongEnvironment(max_life=3, movimiento_px=3)
quantum_agent = QuantumPongAgent(demo_game, discount_factor=0.2, learning_rate=0.01, ratio_explotacion=0.85)

# Entrenamiento
learner, game = play_quantum(rounds=50, learner=quantum_agent, game=demo_game, animate=False)

# Demo final con animación
state = demo_game.reset()
done = False
while not done:
    action = learner.get_next_step(state)
    state, reward, done = demo_game.step(action, animate=True)

q_table = learner.get_q_table(game)
print("Q-table shape:", q_table.shape)
learner.print_q_table(game, digits=2)   # imprime la tabla y la mejor acción por estado
