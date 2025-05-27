# <font color="navy"> Perplexity</font><font color="green">Lab</font>

— *Hi! Listen! I need your help!<br />
Cause' I'm bored again to hell<br />
with the prospect of re-doing,<br />
for this project, the annoying<br />
repeated and painful coding!* 

— *I am here for you,<br /> 
tell me dear friend,<br /> 
what's your conundrum?*

— *What I would really like is vast, <br />
can some pip install magic path <br />
give my code what it needs to have?*
* __Reproducible research__: everyone everywhere should be able to run my code and get the same results. And the main script should be kind of easy to read.
* __File management__: I don't want to waste my time in deciding where to store and get the data and the results.
* __Explore multiple conditions__: I want to execute my experiments/simulations under different conditions specified by my relevant parameters.
* __Parallel__: I want to run several experiments/simulations in parallel without having to rewrite code to do it specifically each time.
* __Remember__: what has been done without worrying about the format nor the place. So I can come later in a year and do old or new analysis and don't straggle with forgotten files and paths. 
* __Avoid re-doing__: automatically check if some experiment was done and load instead of re-doing.
* __Analysis__: once experimentation is done I'd wish to produce with minimal coding some generics or customized plots to analyse the results and get insight on the project.
* __Make reports__: Connect results directly to latex to create a report on the fly. Or directly modify the plots that will be presented without any troublesome file management.
* __Explorable reports__: Create jupyter notebooks with widgets to explore or show results interactively.
* __Carbon and energy footprint__: and why not to get with zero extra effort the accumulated amount of equivalent CO2 and energy consumption of my research to be more conscious and work towards more environmentally responsible research?

— Dear friend, what you need is __PerplexityLab__!
Pipelines Experiments Reproducible Parallel Latex Environmentally conscIous jupYter widgets... Or something around that lines. Anyway, it does that and more! Give it a try!


# Roadmap

- ISSUE: Functions can not be saved -> re run part of experiment that loads corresponding information
- ISSUE: plots receive lists of elements instead of individual ones.
- ISSUE: Multiple nested experiments with nested variables do not work
- Clean stored results: once experiments converged all results do not opened after a script run should be removed.
- Memory limiter: check that the stored results do not exceed certain memory limit.
  - If exceeded then start removing the last accessed results.
- Add energy consumption