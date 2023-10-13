from typing import Any, List, Tuple, Dict, Optional, Type
from yaml import safe_dump
from higgsfield.internal.util import check_name


def build_run_params(params: List["Param"]) -> str:
    return " ".join(param.as_run_param() for param in params)


def build_gh_action_inputs(params: List["Param"]) -> List[str]:
    return [param.as_github_action() for param in params]


_arg_type_set = {
    int,
    float,
    str,
    bool,
}


def reset_values(values: Dict[str, Any]):
    if "name" not in values:
        raise ValueError("Field name is required in Param")
    if "required" not in values:
        values["required"] = False

    return values


class Param:
    name: str
    default: Optional[Any] = None
    description: Optional[str] = None
    required: bool = False
    type: Optional[Type] = None
    options: Optional[Tuple[Any, ...]] = None
    
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    @classmethod
    def from_values(cls, **values) -> "Param":
        values = reset_values(values)
        values["name"] = check_name(values.get("name") or "")
        arg_type = values.get("type")
        default = values.get("default")

        if arg_type is None:
            if default is None:
                arg_type = str
                values["type"] = arg_type
            else:
                arg_type = type(default)
                values["type"] = arg_type

        if arg_type not in _arg_type_set:
            raise ValueError(
                f"Param type {arg_type} not supported."
                + "Only primitive {_arg_type_map.keys()} are supported"
            )

        try:
            if default is not None:
                values["default"] = arg_type(default)
        except Exception as e:
            raise ValueError(
                f"Param default {default} cannot be converted to type {arg_type}"
            ) from e

        if options := values.get("options"):
            values["options"] = [arg_type(opt) for opt in options]
            if default is None:
                values["default"] = values["options"][0]
                default = values["default"]

            if default is not None and default not in values["options"]:
                raise ValueError(f"Param default {default} not in options {options}")

        return Param(**values)

    def check(self, value: str):
        try:
            if self.type is None:
                self.type = str

            value = self.type(value)
        except Exception as e:
            raise ValueError(
                f"Param value {value} cannot be converted to type {self.type}"
            ) from e

        if options := self.options:
            if value not in options:
                raise ValueError(f"Param value {value} not in options {options}")

    def as_github_action(self) -> str:
        indent = "        "
        to_join = [f"{self.name}:"]
        if self.description:
            # TODO: fix that yaml.safe_dump with some proper encoder, string esc etc.
            to_join.append(
                f"{indent}description: {remove_trailing_yaml(safe_dump(self.description))}"
            )
        to_join.append(f"{indent}required: {self.required}")
        if self.default:
            d = self.default
            if type(self.default) == str:
                d = remove_trailing_yaml(safe_dump(self.default))
            to_join.append(f"{indent}default: {d}")
        if self.options:
            to_join.append(f"{indent}options: {list(self.options)}")
            to_join.append(f"{indent}type: choice")
        if self.type == bool:
            to_join.append(f"{indent}type: boolean")
        return "\n".join(to_join)

    def as_run_param(self) -> str:
        # field_name="value"
        return f'{pfx}{self.name}="{wrap_brackets("github.event.inputs." + self.name)}"'

    class Config:
        frozen = True


def wrap_brackets(s: str) -> str:
    left = "${{ "
    right = " }}"
    return left + s + right


def remove_trailing_yaml(s: str) -> str:
    trailing_yaml = "\n...\n"
    if s.endswith(trailing_yaml):
        return s[: -len(trailing_yaml)]
    return s


pfx = "hf_action_"


class _ToSet:
    param: Param
    value: Any

    def __init__(self, param: Param, value: Optional[Any] = None):
        self.param = param
        self.value = value


class ArgParams:
    pass


def parse_kwargs_to_params(
    params: List[Param],
    kwargs: Dict[str, str],
):
    keys = (key[len(pfx) :] for key in kwargs if key.startswith(pfx))

    fields = {param.name: param for param in params}

    # remove keys that are not in subclass_fields
    keys = [key for key in keys if key in fields]

    # get values from kwargs
    values = [kwargs[pfx + key] for key in keys]

    k_v = dict(zip(keys, values))

    params_to_set = {key: _ToSet(param=param) for key, param in fields.items()}

    for key, to_set in params_to_set.items():
        if key in k_v:
            v = k_v[key]
            to_set.param.check(v)
            to_set.value = v
        elif to_set.param.required and to_set.param.default is None:
            raise ValueError(f"Required argument {key} not provided")
        else:
            to_set.value = to_set.param.default

        params_to_set[key] = to_set

    prepare = ArgParams()
    for key, to_set in params_to_set.items():
        setattr(prepare, key, to_set.param.type(to_set.value))

    return prepare
