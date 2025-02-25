# Models from AAMAS'25 Experimental Evaluation
`/discounted` contains models where a discount factor 
of 0.98 has been explicitly unfolded into the model, 
hence we run PHSynt on them. 
`+mem` in folder names in `/discounted` denotes experiment 
where one bit of memory has been explicitly unfolded into the model.

`/undiscounted` contains the same models
without the unfolding of the discount factor. 
We run Inf-Jesp on them, with a discount factor of 
0.98 * 0.98 = 0.9604 (recall that Inf-Jesp supports only two agents).
Each experiment folder contains:
* the original MDP in PRISM syntax(`/sketch.templ`);
* the specification of the hyperproperty in 
  a custom syntax described below(`/sketch.props`); 
* a zip file containing a Cassandra model representing
the cross-product (what we call D in the paper) (`/model.zip`);
* a json file with some necessary information to build a 
  Dec-POMDP for Inf-JESP from the Cassandra model(`/helpers.json`): 
  * target states for setting reachability rewards;
  * a map from states of the cross-product to states tuples of the individual agents, 
    to set observations for agents;
  * a map from actions of the cross-product to actions tuples of the individual agents, 
    to set available actions for agents and the probability distributions they generate.

## Interpreting PHyperLTL formulae.
In a `/sketch.props` file, 
* the first line specifies policies quantifications; 
* the second line is for state quantifications,
* the third line is for the probability constraint.
* 
Probability expressions follow STORM specification syntax. 
Instead of subscripts (as in the formal syntax of the logic), 
we append state variables to atomic propositions in the inner part of the formula
(and to the reward structure, for `/meeting-rew`)
to bind atomic propositions to the right replica. 

## Note
We could not upload the cross-product models in some cases because, even when zipped, 
they would be too big. We would be happy to share them with anyone interested.

## Reference 
[1] Decentralized Planning Using Probabilistic Hyperproperties. Francesco Pontiggia, Filip Macak, Roman Andriushchenko, Michele Chiari and Milan Ceska.
