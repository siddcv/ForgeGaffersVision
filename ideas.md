# Ideas
- Frequency of the touches. Any time there is a disruption, we count it as a lost possession
- Any time that the ball is not on the screen, the user has lost control and is counted as a lost ball
- Starting baseline can just be that both frequencies of the touches per minute of the left and right foot (5% threshold maybe)
- If everything is working, test out a initial bounding box to test whether baseline can be dynamic
- A disruption is basically when the frequency changes and is not immediately repeated after

Ideas to explore:
- Screen positioning bounding box
- A frame by frame (maybe every 5th) breakdown for a more precise analysis
- Take the difference in frequency between every consecutive X frames (for each foot) and compare it against a set bound. If the difference is more negative than a set threshold (and only for a short while), then it should be a ball loss.
- If frequency increases, not a ball loss. If it decreases and stays down, not a ball loss. Only a ball loss if decreases for a few seconds/ frame (need to define this period), and jumps back up, then a ball loss.
