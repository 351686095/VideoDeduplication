import React from "react";
import PropTypes from "prop-types";
import { makeStyles } from "@material-ui/styles";

const useStyles = makeStyles((theme) => ({
  icon: {
    paddingTop: "100%",
    transform: "translate(0%, 0px)",
  },
  horizontal: {
    position: "absolute",
    height: "23%",
    width: "100%",
    left: 0,
    top: "38.5%",
    backgroundColor: theme.palette.common.black,
  },
  vertical: {
    position: "absolute",
    height: "100%",
    width: "23%",
    top: 0,
    left: "38.5%",
    backgroundColor: theme.palette.common.black,
  },
}));

/**
 * Plus icon.
 */
function PlusIcon(props) {
  // TODO: replace with proper implementation
  const { className } = props;
  const classes = useStyles();
  return (
    <div className={className}>
      <div className={classes.icon}>
        <div className={classes.horizontal} />
        <div className={classes.vertical} />
      </div>
    </div>
  );
}

PlusIcon.propTypes = {
  className: PropTypes.string,
};

export default PlusIcon;
