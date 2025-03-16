#Chess game & AI
Steps to optimization of code:
1. Create Reinforcement Learning setup (env, training script)
2. Add a little depth to rewarding system (positive/negative rewards for taking/losing pieces, board control, king safety, win/lose, stockfish best move comparator)
3. Add pretraining script which uses supervised learning
4. Enhance reward system for reinforcement learning
5. Integrate stockfish for supervised learning


TODOs:
- make arm move smoother
- send/receive 6 servos move input instead of one (eg. from 1,60 to 30,60,60,90,45,40)
- add to python settings board size, margin size and square/cell size for generalised chess board arm movement
