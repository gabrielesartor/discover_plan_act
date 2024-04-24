### "Synthesizing Evolving Symbolic Representations for Autonomous Systems"

 <div align="justify">
This work presents a new architecture implementing an open-ended learning system able to synthesize from scratch its experience into a PPDDL representation and update it over time. 
Without a predefined set of goals and tasks, the system integrates intrinsic motivations to explore the environment in a self-directed way, exploiting the high-level knowledge acquired during its experience. The system explores the environment and iteratively: (a) discover options, (b) explore the
environment using options, (c) abstract the knowledge collected and (d) plan. 
This work proposes an alternative approach to implementing open-ended learning architectures exploiting low-level and high-level representations to extend its own knowledge in a virtuous loop.
</div>
<br>
<div align="center">
<img src="https://github.com/gabrielesartor/discover_plan_act/assets/23081850/fcaf06bc-f970-4319-9200-5eb16f636448" width="600">
</div>

<br>


> [!IMPORTANT]
> This work is based on a modified version of the abstraction procedure implemented in https://github.com/sd-james/skills-to-symbols
> and the Gym environment https://github.com/sd-james/gym-treasure-game. Consequently, these two packages are required to run the system.
