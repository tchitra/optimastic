# Optimastic!
 
Optimastic is a project to that aims to help the originator (Tarun) figure out how one should best structure stochastic optimizers. The goals of this project include:

1. Figure out how to extend parallelization schemes (such as _Hogwild_) to stochastic optimizers with momentum
2. Test out some of the new accelerated stochastic optimizers (such as [Katyusha](http://arxiv.org/abs/1603.05953)) that have theoretical promises
3. Make it easy to perform experiments on new optimizers
4. Make an interface that is thread-safe and easy to adapt to the massively parallel regime
