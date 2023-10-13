from .params import Param
from typing import Callable, Any, List, Tuple, Dict, Optional, Union, Set, Type
from higgsfield.internal.util import check_name
from .ast_parser import Expdec, Paramdec


class InnerWrap:
    param_set: Set[Param]

    def __init__(self, func: Callable[..., None], param_set: Set[Param]):
        self.func = func
        self.param_set = set(param_set)

    def add_param(self, param: Param):
        self.param_set.add(param)


class ExperimentDecorator:
    name: str
    params: List[Param]
    train: Callable[..., None]

    def __init__(self, name: str, *, seed: Optional[int] = None):
        """
        Composable experiment decorator.
        >> @ExperimentDecorator("my_experiment", seed=42)
        >> @ParamDecorator("my_param", default=1, description="My param description")
        >> def train(params):
        >>     pass

        that is equivalent to:
        >> def train(params):
        >>     pass
        >>
        >> train = ParamDecorator("my_param", default=1, description="My param description")(train)
        >> train = ExperimentDecorator("my_experiment")(train)

        Params of the experiment are accessible via train.params.


        """
        name = check_name(name)
        if name in _experiments:
            raise ValueError(f"Experiment with name {name} already exists")

        _experiments[name] = self

        self.name = name
        self.params = list()
        if seed is not None:
            self.params.append(Param.from_values(name="seed", default=seed, type=int))
        else:
            self.params.append(Param.from_values(name="seed", type=int, required=True, default=42))
        self.train = lambda x: print("this shouldn't have been called at all")

    def __call__(self, func: Callable[..., None]) -> Optional[Callable[..., None]] :
        if type(func) == InnerWrap:
            # check if seed is in params:
            if not any(param.name == "seed" for param in func.param_set):
                # add seed param
                self.params.extend(list(func.param_set))
            else:
                self.params = list(func.param_set)
            self.train = func.func

            return

        if callable(func) and len(func.__code__.co_varnames) >= 1:
            self.train = func

            return
        else:
            raise ValueError(
                f"Experiment decorator can only be applied to a function that accepts one argument, "
                + f"or on top of another param decorator. \n\tHave: \t{self.name} \n\tFunc: \t{self.train}"
            )

    @classmethod
    def from_ast(
        cls, ast_exps: List[Tuple[Expdec, Dict[str, Paramdec]]]
    ) -> Dict[str, "ExperimentDecorator"]:
        experiments = {}
        for ast_exp, ast_params in ast_exps:
            exp = cls(ast_exp.arg_pairs["name"])
            for ast_param in ast_params.values():
                param = ParamDecorator.from_ast(ast_param)
                exp.params.append(param.param)
            if exp.name in experiments:
                raise ValueError(f"Experiment with name {exp.name} already exists")
            experiments[exp.name] = exp

        return experiments


class ParamDecorator:
    param: Param

    def __init__(
        self,
        name: str,
        *,
        default: Any  = None,
        description: Optional[str] = None,
        required: bool = False,
        type: Optional[Type] = None,
        options: Optional[Union[Tuple[Any, ...], List[Any]]] = None,
    ):
        self.param = Param.from_values(
            name=name,
            default=default,
            description=description,
            required=required,
            type=type,
            options=tuple(options) if options is not None else None,
        )

    def __call__(self, func: Callable[..., None]) -> Callable:
        if type(func) == InnerWrap:
            func.add_param(self.param)
            return func

        if callable(func) and len(func.__code__.co_varnames) >= 1:
            params = set()
            params.add(self.param)
            return InnerWrap(func, params)  # type: ignore
        else:
            raise ValueError(
                "Param decorator can only be applied to a function that accepts one argument, or on top of another param decorator."
            )

    @classmethod
    def list_from_ast(cls, ast_params: Dict[str, Paramdec]):
        params = list()
        for ast_param in ast_params.values():
            param = ParamDecorator.from_ast(ast_param)
            params.append(param)
        return params

    @classmethod
    def from_ast(cls, ast_param: Paramdec):
        return cls(**ast_param.arg_pairs)


_experiments: Dict[str, ExperimentDecorator] = {}


experiment = ExperimentDecorator
param = ParamDecorator
