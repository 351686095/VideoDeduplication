import React from "react";
import clsx from "clsx";
import { makeStyles } from "@material-ui/styles";
import { Theme } from "@material-ui/core";
import Button from "../Button";

const useStyles = makeStyles<Theme>((theme) => ({
  actionsRoot: {
    display: "flex",
    alignItems: "center",
  },
  actionButton: {
    color: theme.palette.common.white,
    "&:hover": {
      backgroundColor: "rgba(255,255,255,0.2)",
    },
    marginLeft: theme.spacing(0.5),
    marginRight: theme.spacing(0.5),
  },
  dark: {
    backgroundColor: "rgba(5,5,5,0.4)",
  },
}));

function invoke(action) {
  return (event) => {
    event.stopPropagation();
    action.handler();
  };
}

function PreviewActions(props: PreviewActionsProps): JSX.Element | null {
  const { actions = [], size = "medium", dark = false, className } = props;
  const classes = useStyles();

  if (actions.length === 0) {
    return null;
  }

  return (
    <div className={clsx(classes.actionsRoot, className)}>
      {actions.map((action) => (
        <Button
          key={action.name}
          onClick={invoke(action)}
          className={clsx(classes.actionButton, dark && classes.dark)}
          size={size}
        >
          {action.name}
        </Button>
      ))}
    </div>
  );
}

type PreviewActionsProps = {
  actions?: {
    name: string;
    handler: (...args: any[]) => void;
  }[];
  size?: "small" | "medium" | "large";
  dark?: boolean;
  className?: string;
};
export default PreviewActions;
