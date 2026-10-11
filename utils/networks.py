from typing import Any, Sequence

import flax.linen as nn
import jax.numpy as jnp


def default_init(scale=1.0):
    """Default kernel initializer."""
    return nn.initializers.variance_scaling(scale, 'fan_avg', 'uniform')


def ensemblize(cls, num_qs, in_axes=None, out_axes=0, **kwargs):
    """Ensemblize a module."""
    return nn.vmap(
        cls,
        variable_axes={'params': 0, 'intermediates': 0},
        split_rngs={'params': True},
        in_axes=in_axes,
        out_axes=out_axes,
        axis_size=num_qs,
        **kwargs,
    )


class MLP(nn.Module):
    """Multi-layer perceptron (MLP).

    Attributes:
        hidden_dims: Hidden layer dimensions.
        activations: Activation function.
        activate_final: Whether to apply activation to the final layer.
        kernel_init: Kernel initializer.
        layer_norm: Whether to apply layer normalization.
    """

    hidden_dims: Sequence[int]
    activations: Any = nn.gelu
    activate_final: bool = False
    kernel_init: Any = default_init()
    layer_norm: bool = False

    @nn.compact
    def __call__(self, x):
        for i, size in enumerate(self.hidden_dims):
            x = nn.Dense(size, kernel_init=self.kernel_init)(x)
            if i + 1 < len(self.hidden_dims) or self.activate_final:
                x = self.activations(x)
                if self.layer_norm:
                    x = nn.LayerNorm()(x)
            if i == len(self.hidden_dims) - 2:
                self.sow('intermediates', 'feature', x)
        return x


class GCValue(nn.Module):
    """Goal-conditioned value/critic function.

    This module can be used for both value V(s, g) and critic Q(s, a, g) functions.

    Attributes:
        hidden_dims: Hidden layer dimensions.
        output_dim: Output dimension (set to None for scalar output).
        mlp_class: MLP class.
        layer_norm: Whether to apply layer normalization.
        num_ensembles: Number of ensemble components.
        gc_encoder: Optional GCEncoder module to encode the inputs.
    """

    hidden_dims: Sequence[int]
    output_dim: int = None
    mlp_class: Any = MLP
    layer_norm: bool = True
    num_ensembles: int = 2
    gc_encoder: nn.Module = None

    def setup(self):
        mlp_class = self.mlp_class
        if self.num_ensembles > 1:
            mlp_class = ensemblize(mlp_class, self.num_ensembles)
        output_dim = self.output_dim if self.output_dim is not None else 1
        value_net = mlp_class((*self.hidden_dims, output_dim), activate_final=False, layer_norm=self.layer_norm)

        self.value_net = value_net

    def __call__(self, observations, goals=None, actions=None, lengths=None, is_encoded=False, goal_encoded=False):
        """Return the value/critic function.

        Args:
            observations: Observations.
            goals: Goals (optional).
            actions: Actions (optional).
            lengths: Lengths (optional).
            is_encoded: Whether the observations/goals are already encoded.
            goal_encoded: Whether goals are already encoded (observations still need encoding).
        """
        if not is_encoded and self.gc_encoder is not None:
            if goals is None:
                inputs = [self.gc_encoder(observations)]
            else:
                inputs = [self.gc_encoder(observations, goals, goal_encoded=goal_encoded)]
        else:
            inputs = [observations]
            if goals is not None:
                inputs.append(goals)
        if actions is not None:
            inputs.append(actions)
        if lengths is not None:
            l = jnp.asarray(lengths)
            if l.ndim == inputs[0].ndim - 1:
                l = jnp.expand_dims(l, -1)
            elif l.ndim != inputs[0].ndim:
                l = jnp.reshape(l, (*l.shape, 1))
            inputs.append(l.astype(jnp.float32))
        inputs = jnp.concatenate(inputs, axis=-1)

        v = self.value_net(inputs)
        if self.output_dim is None:
            v = v.squeeze(-1)

        return v


class ActorVectorField(nn.Module):
    """Actor vector field for flow policies.

    Attributes:
        hidden_dims: Hidden layer dimensions.
        action_dim: Action dimension.
        mlp_class: MLP class.
        layer_norm: Whether to apply layer normalization.
        gc_encoder: Optional GCEncoder module to encode the inputs.
    """

    hidden_dims: Sequence[int]
    action_dim: int
    mlp_class: Any = MLP
    layer_norm: bool = False
    gc_encoder: nn.Module = None

    def setup(self) -> None:
        self.mlp = self.mlp_class(
            (*self.hidden_dims, self.action_dim), activate_final=False, layer_norm=self.layer_norm
        )

    @nn.compact
    def __call__(self, observations, goals=None, actions=None, times=None, is_encoded=False, goal_encoded=False):
        """Return the current vector.

        Args:
            observations: Observations.
            goals: Goals (optional).
            actions: Current actions.
            times: Current times (optional).
            is_encoded: Whether the inputs are already encoded.
            goal_encoded: Whether goals are already encoded (observations still need encoding).
        """
        if not is_encoded and self.gc_encoder is not None:
            if goals is None:
                inputs = self.gc_encoder(observations)
            else:
                inputs = self.gc_encoder(observations, goals, goal_encoded=goal_encoded)
        else:
            if goals is None:
                inputs = observations
            else:
                inputs = jnp.concatenate([observations, goals], axis=-1)
        if times is None:
            inputs = jnp.concatenate([inputs, actions], axis=-1)
        else:
            inputs = jnp.concatenate([inputs, actions, times], axis=-1)

        v = self.mlp(inputs)

        return v
