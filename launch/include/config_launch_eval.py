# Roslaunch eval expression: resolve a dotted key from a RANGE_AID YAML file.
# It must remain a single expression because roslaunch evaluates it inline.
(
    lambda config_file, key: __import__("functools").reduce(
        lambda current, part: current[part],
        str(key).split("."),
        __import__("yaml").safe_load(open(config_file)) or {},
    )
)
