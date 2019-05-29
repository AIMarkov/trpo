from textworld.envs import FrotzEnvironment
import textworld
options=textworld.GameOptions()
options.seeds=1234
options.nb_objects=5
options.quest_length=2
game_file,_=textworld.make(options)
env=textworld.start(game_file)
game_state=env.reset()
env.render()
command="open door"
game_state,reward,done=env.step(command)
env.render()