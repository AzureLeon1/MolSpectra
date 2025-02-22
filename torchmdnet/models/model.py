import re
from typing import Optional, List, Tuple
import torch
from torch.autograd import grad
from torch import nn
from torch_scatter import scatter
from pytorch_lightning.utilities import rank_zero_warn
from torchmdnet.models import output_modules
from torchmdnet.models.wrappers import AtomFilter
from torchmdnet import priors
import warnings


def create_model(args, prior_model=None, mean=None, std=None):
    shared_args = dict(
        hidden_channels=args["embedding_dimension"],
        num_layers=args["num_layers"],
        num_rbf=args["num_rbf"],
        rbf_type=args["rbf_type"],
        trainable_rbf=args["trainable_rbf"],
        activation=args["activation"],
        neighbor_embedding=args["neighbor_embedding"],
        cutoff_lower=args["cutoff_lower"],
        cutoff_upper=args["cutoff_upper"],
        max_z=args["max_z"],
        max_num_neighbors=args["max_num_neighbors"],
    )

    # representation network
    if args["model"] == "graph-network":
        from torchmdnet.models.torchmd_gn import TorchMD_GN

        is_equivariant = False
        representation_model = TorchMD_GN(
            num_filters=args["embedding_dimension"], aggr=args["aggr"], **shared_args
        )
    elif args["model"] == "transformer":
        from torchmdnet.models.torchmd_t import TorchMD_T

        is_equivariant = False
        representation_model = TorchMD_T(
            attn_activation=args["attn_activation"],
            num_heads=args["num_heads"],
            distance_influence=args["distance_influence"],
            **shared_args,
        )
    elif args["model"] == "equivariant-transformer":
        from torchmdnet.models.torchmd_et import TorchMD_ET

        is_equivariant = True
        representation_model = TorchMD_ET(
            attn_activation=args["attn_activation"],
            num_heads=args["num_heads"],
            distance_influence=args["distance_influence"],
            layernorm_on_vec=args["layernorm_on_vec"],
            use_dataset_md17=args["use_dataset_md17"],
            **shared_args,
        )
    else:
        raise ValueError(f'Unknown architecture: {args["model"]}')

    representation_spec_model = None
    if args["uv_model"] == "CNN-AM":
        from torchmdnet.models import CNN_AM

        input_dim = 1500
        in_channel = 1
        representation_spec_model = CNN_AM(
            input_dim=input_dim,
            in_channel=in_channel,
            output_channel=args["embedding_dimension"],
        )
    elif args["uv_model"] == "SpecFormer":
        from torchmdnet.models import SpecFormer

        representation_spec_model = SpecFormer(
            patch_len=args["patch_len"],
            stride=args["stride"],
            output_dim=args["embedding_dimension"],
            input_norm_type=args["input_data_norm_type"],
            # n_heads=n_heads,
            # n_layers=n_layers,
        )

    # atom filter
    if not args["derivative"] and args["atom_filter"] > -1:
        representation_model = AtomFilter(representation_model, args["atom_filter"])
    elif args["atom_filter"] > -1:
        raise ValueError("Derivative and atom filter can't be used together")

    # prior model
    if args["prior_model"] and prior_model is None:
        assert "prior_args" in args, (
            f"Requested prior model {args['prior_model']} but the "
            f'arguments are lacking the key "prior_args".'
        )
        assert hasattr(priors, args["prior_model"]), (
            f'Unknown prior model {args["prior_model"]}. '
            f'Available models are {", ".join(priors.__all__)}'
        )
        # instantiate prior model if it was not passed to create_model (i.e. when loading a model)
        prior_model = getattr(priors, args["prior_model"])(**args["prior_args"])

    # create output network
    output_prefix = "Equivariant" if is_equivariant else ""
    output_model = getattr(output_modules, output_prefix + args["output_model"])(
        args["embedding_dimension"], args["activation"]
    )

    # create the denoising output network
    output_model_noise = None
    if args['output_model_noise'] is not None:
        output_model_noise = getattr(output_modules, output_prefix + args["output_model_noise"])(
            args["embedding_dimension"], args["activation"],
        )

    # create the spec feature output network
    output_model_spec = None
    if args['output_model_spec'] is not None:
        output_model_spec = getattr(output_modules, output_prefix + args["output_model_spec"])(
            args["embedding_dimension"], args["activation"],
        )

    # create the mol feature output network
    output_model_mol = None
    if args['output_model_mol'] is not None:
        output_model_mol = getattr(output_modules, output_prefix + args["output_model_mol"])(
            args["embedding_dimension"], args["activation"],
        )

    # combine representation and output network
    model = TorchMD_Net(
        representation_model,
        output_model,
        prior_model=prior_model,
        reduce_op=args["reduce_op"],
        mean=mean,
        std=std,
        derivative=args["derivative"],
        output_model_noise=output_model_noise,
        position_noise_scale=args["position_noise_scale"],
        representation_spec_model=representation_spec_model,
        output_model_spec=output_model_spec,
        output_model_mol=output_model_mol,
    )
    return model


def load_model(filepath, args=None, device="cpu", mean=None, std=None, **kwargs):
    ckpt = torch.load(filepath, map_location="cpu")
    if args is None:
        args = ckpt["hyper_parameters"]

    for key, value in kwargs.items():
        if not key in args:
            warnings.warn(f'Unknown hyperparameter: {key}={value}')
        args[key] = value

    model = create_model(args)

    state_dict = {re.sub(r"^model\.", "", k): v for k, v in ckpt["state_dict"].items()}

    # NOTE for debug
    new_state_dict = {}
    for k, v in state_dict.items():
        # if 'pos_normalizer' not in k:
        if "output_model_noise.0" in k:
            k = k.replace("output_model_noise.0", "output_model_noise")
        if "head.2" in k:
            continue
        new_state_dict[k] = v

    current_model_dict = model.state_dict()
    # ommit mismatching shape
    new_state_dict2 = {}
    for k in current_model_dict:
        if k in new_state_dict:
            # print(k, current_model_dict[k].size(), new_state_dict[k].size())
            if current_model_dict[k].size() == new_state_dict[k].size():
                new_state_dict2[k] = new_state_dict[k]
            else:
                print(f"warning {k} shape mismatching, not loaded")
                new_state_dict2[k] = current_model_dict[k]

    # loading_return = model.load_state_dict(state_dict, strict=False)
    loading_return = model.load_state_dict(new_state_dict2, strict=False)

    if len(loading_return.unexpected_keys) > 0:
        # Should only happen if not applying denoising during fine-tuning.
        assert all(
            (
                "output_model_noise" in k
                or "pos_normalizer" in k
                or "representation_spec_model" in k
                or "output_model_spec" in k
                or "output_model_mol" in k
            )
            for k in loading_return.unexpected_keys
        )
    # assert len(loading_return.missing_keys) == 0, f"Missing keys: {loading_return.missing_keys}"

    if mean:
        model.mean = mean
    if std:
        model.std = std

    return model.to(device)


class TorchMD_Net(nn.Module):

    def __init__(
        self,
        representation_model,
        output_model,
        prior_model=None,
        reduce_op="add",
        mean=None,
        std=None,
        derivative=False,
        output_model_noise=None,
        position_noise_scale=0.0,
        representation_spec_model=None,
        output_model_spec=None,
        output_model_mol=None,
    ):
        super(TorchMD_Net, self).__init__()
        self.representation_model = representation_model
        self.output_model = output_model
        self.representation_spec_model = representation_spec_model

        self.prior_model = prior_model
        if not output_model.allow_prior_model and prior_model is not None:
            self.prior_model = None
            rank_zero_warn(
                (
                    "Prior model was given but the output model does "
                    "not allow prior models. Dropping the prior model."
                )
            )

        self.reduce_op = reduce_op
        self.derivative = derivative
        self.output_model_noise = output_model_noise        
        self.position_noise_scale = position_noise_scale

        self.output_model_spec = output_model_spec
        self.output_model_mol = output_model_mol

        mean = torch.scalar_tensor(0) if mean is None else mean
        self.register_buffer("mean", mean)
        std = torch.scalar_tensor(1) if std is None else std
        self.register_buffer("std", std)

        if self.position_noise_scale > 0:
            self.pos_normalizer = AccumulatedNormalization(accumulator_shape=(3,))

        self.reset_parameters()

    def reset_parameters(self):
        self.representation_model.reset_parameters()
        self.output_model.reset_parameters()
        if self.output_model_noise is not None:
            self.output_model_noise.reset_parameters()
        if self.representation_spec_model is not None:
            self.representation_spec_model.reset_parameters()
        if self.output_model_spec is not None:
            self.output_model_spec.reset_parameters()
        if self.output_model_mol is not None:
            self.output_model_mol.reset_parameters()
        if self.prior_model is not None:
            self.prior_model.reset_parameters()

    def forward(self, z, pos, spec_list, batch: Optional[torch.Tensor] = None):
        assert z.dim() == 1 and z.dtype == torch.long
        batch = torch.zeros_like(z) if batch is None else batch

        if self.derivative:
            pos.requires_grad_(True)

        # run the potentially wrapped representation model
        x, v, z, pos, batch = self.representation_model(z, pos, batch=batch)

        # construct spectra feature
        spec_feature = None
        loss_reconstruct = None
        if self.representation_spec_model is not None:
            if spec_list is not None:
                spec_feature = self.representation_spec_model(spec_list)
            if spec_feature is not None and len(spec_feature) == 2:
                spec_feature, loss_reconstruct = spec_feature

        # construct molecule feature
        mol_feature = scatter(x, batch, dim=0, reduce=self.reduce_op)
        if self.output_model_mol is not None:
            mol_feature = self.output_model_mol.pre_reduce(x, v, z, pos, batch)
            mol_feature = scatter(mol_feature, batch, dim=0, reduce=self.reduce_op)

        # predict noise
        noise_pred = None
        if self.output_model_noise is not None:
            noise_pred = self.output_model_noise.pre_reduce(x, v, z, pos, batch) 

        # apply the output network
        x = self.output_model.pre_reduce(x, v, z, pos, batch)

        # scale by data standard deviation
        if self.std is not None:
            x = x * self.std

        # apply prior model
        if self.prior_model is not None:
            x = self.prior_model(x, z, pos, batch)

        # aggregate atoms
        out = scatter(x, batch, dim=0, reduce=self.reduce_op)

        # shift by data mean
        if self.mean is not None:
            out = out + self.mean

        # apply output model after reduction
        out = self.output_model.post_reduce(out)

        # compute gradients with respect to coordinates
        if self.derivative:
            grad_outputs: List[Optional[torch.Tensor]] = [torch.ones_like(out)]
            dy = grad(
                [out],
                [pos],
                grad_outputs=grad_outputs,
                create_graph=True,
                retain_graph=True,
            )[0]
            if dy is None:
                raise RuntimeError("Autograd returned None for the force prediction.")
            return out, noise_pred, -dy, spec_feature, mol_feature, loss_reconstruct
        # TODO: return only `out` once Union typing works with TorchScript (https://github.com/pytorch/pytorch/pull/53180)
        return out, noise_pred, None, spec_feature, mol_feature, loss_reconstruct


class AccumulatedNormalization(nn.Module):
    """Running normalization of a tensor."""
    def __init__(self, accumulator_shape: Tuple[int, ...], epsilon: float = 1e-8):
        super(AccumulatedNormalization, self).__init__()

        self._epsilon = epsilon
        self.register_buffer("acc_sum", torch.zeros(accumulator_shape))
        self.register_buffer("acc_squared_sum", torch.zeros(accumulator_shape))
        self.register_buffer("acc_count", torch.zeros((1,)))
        self.register_buffer("num_accumulations", torch.zeros((1,)))

    def update_statistics(self, batch: torch.Tensor):
        batch_size = batch.shape[0]
        self.acc_sum += batch.sum(dim=0)
        self.acc_squared_sum += batch.pow(2).sum(dim=0)
        self.acc_count += batch_size
        self.num_accumulations += 1

    @property
    def acc_count_safe(self):
        return self.acc_count.clamp(min=1)

    @property
    def mean(self):
        return self.acc_sum / self.acc_count_safe

    @property
    def std(self):
        return torch.sqrt(
            (self.acc_squared_sum / self.acc_count_safe) - self.mean.pow(2)
        ).clamp(min=self._epsilon)

    def forward(self, batch: torch.Tensor):
        if self.training:
            self.update_statistics(batch)
        return ((batch - self.mean) / self.std)
