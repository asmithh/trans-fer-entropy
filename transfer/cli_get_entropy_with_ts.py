import sys, argparse, logging
from symbolic_transfer_entropy import symbolic_transfer_entropy

# specify topic, state, time lag, etc. and structure it/either command line script or config


def main(args, loglevel):
    """This is the main function to calculate the trans-entropy."""
    logging.basicConfig(
        format="%(asctime)s %(name)-12s %(levelname)-8s %(message)s",
        level=logging.DEBUG,
    )

    logging.debug(args)

    master_file = pd.read_pickle(args.f)

    x_level = args.x_geographical_level
    y_level = args.y_geographical_level
    x_metric = args.x_metric
    y_metric = args.y_metric
    value_col = args.value_col

    # unknown the col name for distinguishing CA, NA or others. Call this geo for now. We could also specify an arg for this?
    x = master_file.loc[
        (master_file["geo"] == x_level) & (master_file["metric"] == x_metric), value_col
    ]
    y = master_file.loc[
        (master_file["geo"] == y_level) & (master_file["metric"] == y_metric), value_col
    ]

    # assume there is a col for date? do we need to sort it first?

    w = args.w
    s = args.s

    # calc x on y
    scores_x_on_y = []
    x_on_y.append((kw, symbolic_transfer_entropy(y, x, w, s)))

    # saving results. do we need an arg for the path that saves the results?
    with open(f"{x_level}_on_{y_level})_{x_metric}_{y_metric}_w{w}_s{s}.txt", "w") as f:
        for line in sorted(scores_x_on_y, key=lambda x: x[1], reverse=True)[0:200]:
            f.write(",".join([str(l) for l in line]) + "\n")
    # if bilaterial is true calc y on x
    if args.b:
        scores_y_on_x = []
        y_on_x.append((kw, symbolic_transfer_entropy(x, y, w, s)))

        # saving results. do we need an arg for the path that saves the results?
        with open(
            f"{y_level}_on_{x_level})_{y_metric}_{x_metric}_w{w}_s{s}.txt", "w"
        ) as f:
            for line in sorted(scores_y_on_x, key=lambda x: x[1], reverse=True)[0:200]:
                f.write(",".join([str(l) for l in line]) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="TRANSFER ENTROPY WITH TIME SERIES!",
        epilog="The ideal input will be a long table in .pkl format, and with identifiers to slice a time series of various kinds.\
            It needs to have columns like 'geo' to indicate the state/national levels, 'metric' to indicate what kind of \
                time series is examined, e.g. 'keyword' or 'topics'. There should be a value column to store\
                    all the numeric values for time series. ",
        fromfile_prefix_chars="@",
    )

    # 0. Path to load time series
    parser.add_argument(
        "-f",
        "--file_path",
        dest="f",
        type=str,
        help="Specify path of the file to read the time series data. Input should be string",
    )

    # 1. Specifying the ways to identify X and Y time series
    # 1.1.1 Specify the geo level for X
    parser.add_argument(
        "-xg",
        "--x_geographical_level",
        # "x_geographical_level",
        type=str,
        dest="x_geographical_level",
        help="Specify Geographical Level for X, e.g. TX.",
    )

    # 1.1.2 Specify the geo level for Y
    parser.add_argument(
        "-yg",
        "--y_geographical_level",
        dest="y_geographical_level",
        type=str,
        help="Specify Geographical Level for Y, e.g. TX.",
    )

    # 1.2.1  Specify the specific metric for X
    parser.add_argument(
        "-xm",
        "--x_metric",
        dest="x_metric",
        type=str,
        help="Specify the metric to use for X, e.g. count.",
    )

    # 1.2.2 Specify the geo level for Y
    parser.add_argument(
        "-ym",
        "--y_metric",
        dest="y_metric",
        type=str,
        help="Specify the metric to use for Y, e.g. count.",
    )

    # 1.3 The column name to get the value for the time series
    parser.add_argument(
        "-v",
        "--value_col",
        type=str,
        default="value",
        dest="value_col",
        help="Specify the column to use for getting the values for X and Y, e.g. value.",
    )

    # 2. Specify values for the trans-entropy_config
    # 2.1 sliding window time
    parser.add_argument(
        "-s",
        "--sliding_window",
        default=4,
        dest="s",
        type=int,
        help="Specify the symbol window length. The input should be an int.",
    )

    # 2.2 sliding window
    parser.add_argument(
        "-w",
        "--window_length",
        type=int,
        default=1,
        dest="w",
        help="Specify the sliding window time. The input should be an int",
        # required=True,
    )

    # 2.3 bilateral
    parser.add_argument(
        "-b",
        "--bilateral",
        type=bool,
        default=True,
        dest="b",
        help="If True, will return both x on y and y on x.",
        # required=True,
    )

    args = parser.parse_args()

    print(args)

    if args.f is None:
        raise ValueError("File path empty")

    main(args)
