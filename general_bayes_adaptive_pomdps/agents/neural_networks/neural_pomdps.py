""" POMDP dynamics as neural networks """

import abc
from collections import deque, namedtuple
from enum import Enum, auto
from typing import Any, Deque, List, Tuple, Optional

import numpy as np
import torch
import torch.distributions.utils
from typing_extensions import Protocol

from general_bayes_adaptive_pomdps.agents.neural_networks import Net
from general_bayes_adaptive_pomdps.agents.neural_networks.misc import perturb
from general_bayes_adaptive_pomdps.environments import ActionSpace
from general_bayes_adaptive_pomdps.misc import DiscreteSpace
from general_bayes_adaptive_pomdps.pytorch_api import device, log_tensorboard


class Interaction(namedtuple("interaction", "state action next_state observation")):
    """transition in environment

    Contains:
         state: (`np.ndarray`):
         action: (`int`):
         next_state: (`np.ndarray`):
         observation: (`np.ndarray`):
    """

    __slots__ = ()  # required to keep lightweight implementation of namedtuple


class OptimizerBuilder(Protocol):
    """ Defines the signature of optimizer builder"""

    def __call__(
        self, parameters: Any, learning_rate: float
    ) -> torch.optim.Optimizer:  # type: ignore
        """function call signature for building an optimizer

        Args:
             parameters: (torch parameters): the parameters to optimize
             learning_rate: (`float`): learning rate of the optimizer

        RETURNS (`torch.optim.Optimizer`):

        """


def sgd_builder(parameters: Any, learning_rate: float) -> torch.optim.Optimizer:  # type: ignore
    """builds the torch SGD optimizer to update `parameters` with `learning_rate` stepsize

    Args:
         parameters: (torch parameters): the parameters to optimize
         learning_rate: (`float`): learning rate of the optimizer

    RETURNS (`torch.optim.Optimizer`): Torch SGD optimizer

    """
    return torch.optim.SGD(parameters, lr=learning_rate)


def adam_builder(
    parameters: Any, learning_rate: float
) -> torch.optim.Optimizer:  # type: ignore
    """builds the torch Adam optimizer to update `parameters` with `learning_rate` stepsize

    Args:
         parameters: (torch parameters): the parameters to optimize
         learning_rate: (`float`): learning rate of the optimizer

    RETURNS (`torch.optim.Optimizer`): Torch Adam optimizer

    """
    return torch.optim.Adam(parameters, lr=learning_rate)


def get_optimizer_builder(option: str) -> OptimizerBuilder:
    """Returns the appropriate optimizer builder

    Args:
         option: (`str`): in ['SGD', 'Adam']

    RETURNS (`general_bayes_adaptive_pomdps.agents.neural_networks.OptimizerBuilder`):

    """
    if option == "SGD":
        return sgd_builder
    if option == "Adam":
        return adam_builder

    raise ValueError(f"Undefined optimizer {option}")


class DynamicsModel:
    """ POMDP dynamics (s,a) -> p(s',o) """

    @staticmethod
    def sample_from_model(model: List[np.ndarray], num: int) -> np.ndarray:
        """samples `num` instances from model

        The model is a list of categorical distributions, where each element
        represents a dimension or feature. The element itself is assumed to be
        a proper distribution (sum up to one).

        Args:
            model (`List[np.ndarray]`): model[i][j] is probability of feature i being j
            num (`int`): number of samples

        Returns:
            `np.ndarray`: list of samples, ith element is a sample with
            len(model) features
        """

        # sample value for each dimension iteratively
        return np.array(
            [
                torch.multinomial(torch.from_numpy(probs), num, replacement=True).item()
                for probs in model
            ],
            dtype=int,
        )

    class T(abc.ABC):
        """interface for a transition model in `DynamicsModel`"""

        @abc.abstractmethod
        def model(self, state: np.ndarray, action: int) -> List[np.ndarray]:
            """next-state distribution model given state-action pair

            Args:
                state (`np.ndarray`):
                action (`int`):

            Returns:
                `List[np.ndarray]`: model, [i][j] is probability of feature i
                taking value j
            """

        @abc.abstractmethod
        def sample(
            self,
            state: np.ndarray,
            action: int,
            num: int,
        ) -> np.ndarray:
            """sample `num` state given `state`-`action` pair

            can be implemented by calling
            `sample_from_model(self.model(state, action, num))`

            Args:
                state (`np.ndarray`):
                action (`int`):
                num (`int`): number of samples to provide

            Returns:
                `np.ndarray`: `num` states
            """

    class ObsModel(abc.ABC):
        """interface for an observation model in `DynamicsModel`"""

        @abc.abstractmethod
        def model(
            self, state: np.ndarray, action: int, next_state: np.ndarray
        ) -> List[np.ndarray]:
            """observation distribution model given `state`-`action`-`next_state` triplet

            Args:
                state (`np.ndarray`):
                action (`int`):
                next_state (`np.ndarray`):

            Returns:
                `List[np.ndarray]`: model, [i][j] is probability of feature i
                taking value j
            """

        @abc.abstractmethod
        def sample(
            self,
            state: np.ndarray,
            action: int,
            next_state: np.ndarray,
            num: int,
        ) -> np.ndarray:
            """sample `num` observation given `state`-`action`-`next_state` triplet

            can be implemented by calling
            `sample_from_model(self.model(state, action, next_state, num))`

            Args:
                state (`np.ndarray`):
                action (`int`):
                next_state (`np.ndarray`):
                num (`int`): number of samples to provide

            Returns:
                `np.ndarray`: `num` observations
            """

    class NN:
        """A neural network in the `DynamicsModel`, either `O` or `T`"""

        def __init__(
            self,
            net: Net,
            optimizer_builder: OptimizerBuilder,
            learning_rate: float,
        ):
            self.criterion = torch.nn.CrossEntropyLoss()
            self.net = net

            self._learning_rate = learning_rate
            self._optimizer_builder = optimizer_builder
            self._optimizer = self._optimizer_builder(
                net.parameters(), self._learning_rate
            )

        def reset(self) -> None:
            """reset the network (randomly) and optimizer """
            self.net.random_init_parameters()
            self._optimizer = self._optimizer_builder(
                self.net.parameters(), self._learning_rate
            )

        def set_learning_rate(self, learning_rate: float) -> None:
            """(re)sets the optimizers' learning rate

            Will re-create the optimizers, thus losing its current state

            Args:
                 learning_rate: (`float`):

            RETURNS (`None`):

            """
            assert (
                0 <= learning_rate < 1
            ), f"learning rate must be [0,1], not {learning_rate}"
            self._learning_rate = learning_rate

            self._optimizer = self._optimizer_builder(
                self.net.parameters(), self._learning_rate
            )

        def perturb(
            self,
            stdev: float = 0.1,
        ) -> None:
            """perturb parameters of model

            Args:
                 stdev: (`float`): the standard deviation of the perturbation

            RETURNS (`None`):

            """
            with torch.no_grad():
                for param in self.net.parameters():
                    param.set_(perturb(param, stdev))

    class TNet(NN, T):
        """Neural Network implementation of `T` in `DynamicsModel`"""

        def __init__(
            self,
            state_space: DiscreteSpace,
            action_space: ActionSpace,
            optimizer_builder: OptimizerBuilder,
            learning_rate: float,
            network_size: int,
            dropout_rate: float,
            input_state_size: Optional[int] = None,
        ):
            """Initiates a neural network transition model

            This model implements :class:`T` with the help of :class:`NN`.

            ``state_space`` and ``action_space`` are used as ways to figure out
            the size of the model. Note that with ``input_state_size`` the
            input of the state can be overwritten. The idea here is that it
            should be possible to provide one-hot encoding or in general an
            input that is different from the state space. If that parameter is
            not given, we assume the number of dimensions in ``state_space``
            determine the input of the model.

            Most of the configuration of the network is done through the
            parameters ``learning_rate``, ``network_size``, ``dropout_rate``
            and ``optimizer_builder``. Here ``optimizer_builder`` is a
            constructor that, given the network weights, creates the optimizer.

            :param state_space: the expected shape of the state as in- and output of the model
            :param action_space: determines action input (assumed one-hot)
            :param optimizer_builder: constructs the optimizer given weights
            :param learning_rate: learning rate used by optimizer
            :param network_size: # of nodes in the 2-layer network
            :param dropout_rate: rate of dropping nodes
            :param input_state_size: overwrites the number of input nodes
                (otherwise set to # dimensions in ``state_space``)
            """
            if not input_state_size:
                input_state_size = state_space.ndim

            net = Net(
                input_size=input_state_size + action_space.n,
                output_size=np.sum(state_space.size),
                layer_size=network_size,
                dropout_rate=dropout_rate,
            ).to(device())

            super().__init__(net, optimizer_builder, learning_rate)
            self.action_space = action_space
            self.state_space = state_space

        def batch_train(
            self,
            states: torch.Tensor,
            actions: torch.Tensor,
            next_states: torch.Tensor,
        ) -> float:
            """trains on the given batch

            Args:
                states (`torch.Tensor`): assumed torch to facilitate performance
                actions (`torch.Tensor`): assumed torch to facilitate performance
                next_states (`torch.Tensor`): assumed torch to facilitate performance

            Returns:
                `float`: loss
            """
            state_action_pairs = torch.cat((states, actions), dim=1)
            next_state_logits = self.net(state_action_pairs)

            loss = torch.stack(
                [
                    self.criterion(
                        next_state_logits[
                            :,
                            self.state_space.dim_cumsum[
                                i
                            ] : self.state_space.dim_cumsum[i + 1],
                        ],
                        next_states[:, i],
                    )
                    for i in range(self.state_space.ndim)
                ]
            ).sum()

            self._optimizer.zero_grad()
            loss.backward()
            self._optimizer.step()

            return loss.item()

        def model(self, state: np.ndarray, action: int) -> List[np.ndarray]:
            """`T` interface"""

            state_action_pair = (
                torch.from_numpy(
                    np.concatenate([state, self.action_space.one_hot(action)])
                )
                .to(device())
                .float()
            )

            with torch.no_grad():
                logits = self.net(state_action_pair)

                return [
                    torch.distributions.utils.logits_to_probs(  # type: ignore
                        logits[
                            self.state_space.dim_cumsum[
                                i
                            ] : self.state_space.dim_cumsum[i + 1]
                        ]
                    )
                    .cpu()
                    .numpy()
                    for i in range(self.state_space.ndim)
                ]

        def sample(
            self,
            state: np.ndarray,
            action: int,
            num: int,
        ) -> np.ndarray:
            """`T` interface"""
            transition_model = self.model(state, action)
            return DynamicsModel.sample_from_model(transition_model, num)

    class ONet(NN, ObsModel):
        """Neural Network implementation of `O` in `DynamicsModel`"""

        def __init__(
            self,
            state_space: DiscreteSpace,
            action_space: ActionSpace,
            obs_space: DiscreteSpace,
            optimizer_builder: OptimizerBuilder,
            learning_rate: float,
            network_size: int,
            dropout_rate: float,
        ):

            net = Net(
                input_size=state_space.ndim * 2 + action_space.n,
                output_size=np.sum(obs_space.size),
                layer_size=network_size,
                dropout_rate=dropout_rate,
            ).to(device())

            super().__init__(net, optimizer_builder, learning_rate)
            self.action_space = action_space
            self.obs_space = obs_space

        def batch_train(
            self,
            states: torch.Tensor,
            actions: torch.Tensor,
            next_states: torch.Tensor,
            obs: torch.Tensor,
        ) -> float:
            """trains on the given batch

            Args:
                states (`torch.Tensor`): assumed torch to facilitate performance
                actions (`torch.Tensor`): assumed torch to facilitate performance
                next_states (`torch.Tensor`): assumed torch to facilitate performance
                obs (`torch.Tensor`): assumed torch to facilitate performance

            Returns:
                `float`: loss
            """
            state_action_state_triplets = torch.cat(
                (states, actions, next_states.float()), dim=1
            )
            observation_logits = self.net(state_action_state_triplets)

            loss = torch.stack(
                [
                    self.criterion(
                        observation_logits[
                            :,
                            self.obs_space.dim_cumsum[i] : self.obs_space.dim_cumsum[
                                i + 1
                            ],
                        ],
                        obs[:, i],
                    )
                    for i in range(self.obs_space.ndim)
                ]
            ).sum()

            self._optimizer.zero_grad()
            loss.backward()
            self._optimizer.step()

            return loss.item()

        def model(
            self, state: np.ndarray, action: int, next_state: np.ndarray
        ) -> List[float]:
            """`O` interface"""

            state_action_state = (
                torch.from_numpy(
                    np.concatenate(
                        [state, self.action_space.one_hot(action), next_state]
                    )
                )
                .to(device())
                .float()
            )

            with torch.no_grad():
                logits = self.net(state_action_state)

                return [
                    torch.distributions.utils.logits_to_probs(  # type: ignore
                        logits[
                            self.obs_space.dim_cumsum[i] : self.obs_space.dim_cumsum[
                                i + 1
                            ]
                        ]
                    )
                    .cpu()
                    .numpy()
                    for i in range(self.obs_space.ndim)
                ]

        def sample(
            self,
            state: np.ndarray,
            action: int,
            next_state: np.ndarray,
            num: int,
        ):
            """`O` interface"""
            obs_model = self.model(state, action, next_state)
            return DynamicsModel.sample_from_model(obs_model, num)

    class FreezeModelSetting(Enum):
        """ setting for training """

        FREEZE_NONE = auto()
        FREEZE_T = auto()
        FREEZE_O = auto()

    def __init__(
        self,
        state_space: DiscreteSpace,
        action_space: ActionSpace,
        batch_size: int,
        t_model: T,
        o_model: ObsModel,
    ):
        """Creates a dynamic model

        Args:
             state_space: (`general_bayes_adaptive_pomdps.misc.DiscreteSpace`):
             action_space: (`general_bayes_adaptive_pomdps.environments.ActionSpace`):
             obs_space: (`general_bayes_adaptive_pomdps.misc.DiscreteSpace`):
             network_size: (`int`): number of nodes in hidden layers
             learning_rate: (`float`): learning rate of the optimizers
             batch_size: (`int`): number of interactions to **remember** and update with
             dropout_rate: (`float`): dropout rate of the layers
             optimizer_builder: (`OptimizerBuilder`): builder function for optimizer

        """

        self.state_space = state_space
        self.action_space = action_space

        self.experiences: Deque[Interaction] = deque([], batch_size)

        self.num_batches = 0

        self.t = t_model
        self.o = o_model

    def set_learning_rate(self, learning_rate: float) -> None:
        """(re)sets the optimizers' learning rate

        Will re-create the optimizers, thus losing its current state

        Args:
             learning_rate: (`float`):

        RETURNS (`None`):

        """

        for model in [self.t, self.o]:
            if isinstance(model, DynamicsModel.NN):
                model.set_learning_rate(learning_rate)

    def simulation_step(
        self, state: np.array, action: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """The simulation step of this dynamics model: S x A -> S, O

        Args:
             state: (`np.array`): input state
             action: (`int`): chosen action

        RETURNS (`Tuple[np.ndarray, np.ndarray]`): [state, observation]

        """

        assert self.state_space.contains(state), f"{state} not in {self.state_space}"
        assert self.action_space.contains(
            action
        ), f"{action} not in {self.action_space}"

        next_state = self.sample_state(state, action)
        observation = self.sample_observation(state, action, next_state)

        return next_state, observation

    def batch_update(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        next_states: np.ndarray,
        obs: np.ndarray,
        log_loss: bool = False,
        conf: FreezeModelSetting = FreezeModelSetting.FREEZE_NONE,
    ) -> None:
        """performs a batch update (single gradient descent step)

        Args:
             states: (`np.ndarray`): (batch_size, state_shape) array of states
             actions: (`np.ndarray`): (batch_size,) array of actions
             next_states: (`np.ndarray`): (batch_size, state_shape) array of (next) states
             obs: (`np.ndarray`): (batch_size, obs_shape) array of observations
             log_loss: (`bool`): whether to log the loss
             conf: (`FreezeModelSetting`): configurations for training

        """

        states = torch.from_numpy(states).float().to(device())
        actions = (
            # pylint: disable=not-callable
            torch.tensor([self.action_space.one_hot(a) for a in actions])
            .float()
            .to(device())
        )
        next_states = torch.from_numpy(next_states).to(device())
        obs = torch.from_numpy(obs).to(device())

        # transition model
        if conf != DynamicsModel.FreezeModelSetting.FREEZE_T and isinstance(
            self.t, DynamicsModel.NN
        ):

            loss = self.t.batch_train(states, actions, next_states)  # type: ignore

            if log_loss:
                log_tensorboard(f"transition_loss/{self}", loss, self.num_batches)

        # observation model
        if conf != DynamicsModel.FreezeModelSetting.FREEZE_O and isinstance(
            self.o, DynamicsModel.NN
        ):

            loss = self.o.batch_train(states, actions, next_states, obs)  # type: ignore

            if log_loss:
                log_tensorboard(f"observation_loss/{self}", loss, self.num_batches)

        self.num_batches += 1

    def self_learn(
        self, conf: FreezeModelSetting = FreezeModelSetting.FREEZE_NONE
    ) -> None:
        """performs a batch update on stored data

        Args:
             conf: (`FreezeModelSetting`): configurations for training

        RETURNS (`None`):

        """
        assert self.experiences, "cannot self learn without data"

        log_loss = False  # no logging of loss online
        self.batch_update(*map(np.array, zip(*self.experiences)), log_loss, conf)  # type: ignore

    def add_transition(
        self,
        state: np.ndarray,
        action: int,
        next_state: np.ndarray,
        observation: np.ndarray,
    ) -> None:
        """stores the given transition

        `this` uses this data to learn

        Args:
             state: (`np.ndarray`):
             action: (`int`):
             next_state: (`np.ndarray`):
             observation: (`np.ndarray`):

        RETURNS (`None`):

        """

        self.experiences.append(Interaction(state, action, next_state, observation))

    def sample_state(self, state: np.ndarray, action: int, num: int = 1) -> np.ndarray:
        """samples next state given current and action

        Args:
             state: (`np.ndarray`): current state
             action: (`int`): taken action
             num: (`int`): number of samples

        RETURNS (`np.ndarray`): next state

        """
        return self.t.sample(state, action, num)

    def sample_observation(
        self,
        state: np.ndarray,
        action: int,
        next_state: np.ndarray,
        num: int = 1,
    ) -> np.ndarray:
        """samples an observation given state - action - next state triple

        Args:
             state: (`np.ndarray`): state at t
             action: (`int`): taken action at t
             next_state: (`np.ndarray`): state t + 1
             num: (`int`): number of samples

        RETURNS (`np.ndarray`): observation at t + 1

        """
        return self.o.sample(state, action, next_state, num)

    def transition_model(self, state: np.ndarray, action: int) -> List[np.ndarray]:
        """Returns the transition model (next state) for state-action pair

        Element i of the returned list is the (batch, dim_size) probabilities
        of dimension i as a numpy array

        Args:
             state: (`np.ndarray`):
             action: (`int`):

        RETURNS (`List[np.ndarray]`): [#dim x dim_size] list

        """

        return self.t.model(state, action)

    def observation_model(
        self, state: np.ndarray, action: int, next_state: np.ndarray
    ) -> List[np.ndarray]:
        """Returns the observation model of a transition

        Element i of the returned list is the probability of observation
        dimension i

        Args:
             state: (`np.ndarray`):
             action: (`int`):
             next_state: (`np.ndarray`):

        RETURNS (`List[np.ndarray]`): [#dim x dim_size] list

        """

        return self.o.model(state, action, next_state)

    def reset(self) -> None:
        """ resets the networks """
        self.experiences.clear()

        for model in [self.t, self.o]:
            if isinstance(model, DynamicsModel.NN):
                model.reset()

        self.num_batches = 0

    def perturb_parameters(
        self,
        stdev: float = 0.1,
        freeze_model_setting: FreezeModelSetting = FreezeModelSetting.FREEZE_NONE,
    ) -> None:
        """perturb parameters of model

        Args:
             stdev: (`float`): the standard deviation of the pertubation
             freeze_model_setting: (`FreezeModelSetting`)

        RETURNS (`None`):

        """

        if (
            freeze_model_setting != DynamicsModel.FreezeModelSetting.FREEZE_T
            and isinstance(self.t, DynamicsModel.NN)
        ):
            self.t.perturb(stdev)

        if (
            freeze_model_setting != DynamicsModel.FreezeModelSetting.FREEZE_O
            and isinstance(self.o, DynamicsModel.NN)
        ):
            self.o.perturb(stdev)
