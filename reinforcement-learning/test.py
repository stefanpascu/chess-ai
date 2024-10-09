from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from chess_env import ChessEnv  # Ensure you're importing the class correctly

# Wrap the ChessEnv in a DummyVecEnv and pass the render mode
vec_env = DummyVecEnv([lambda: ChessEnv(render_mode='human')])  # Set to 'human' or any other mode as needed

# Initialize the PPO model with the policy and environment
model = PPO("MlpPolicy", vec_env, verbose=1)

# Train the model
model.learn(total_timesteps=5000)

# Save the trained model
model.save("chess_model")

# Load and Play a Game
# Initialize a new DummyVecEnv
vec_env = DummyVecEnv([lambda: ChessEnv(render_mode='human')])

# Load the trained model
model = PPO.load("chess_model")

# Reset the VecEnv (this will return just the observation, not a tuple)
obs = vec_env.reset()

# Play a game until it ends
while True:
    action, _states = model.predict(obs)  # Get the action based on the observation
    obs, reward, done, info = vec_env.step(action)  # Step in VecEnv
    vec_env.render()  # Render the board

    if done:
        obs = vec_env.reset()  # Reset the environment for a new game
        print("Game Over")
        break  # Exit the loop if the game is over
