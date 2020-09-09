import React, { useCallback } from "react";
import clsx from "clsx";
import PropTypes from "prop-types";
import { makeStyles } from "@material-ui/styles";
import SelectionDecorator from "./SelectionDecorator";
import { ButtonBase } from "@material-ui/core";
import useUniqueId from "../hooks/useUniqueId";

const useStyles = makeStyles((theme) => ({
  container: {
    flexGrow: 1,
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
  },
  link: {
    padding: "5px 4px 8px 4px",
    cursor: "pointer",
    /**
     * Ensure selection decorator is displayed correctly.
     */
    transform: "translate(0%, 0px)",
  },
  textSelected: {
    fontWeight: "bold",
    color: theme.palette.primary.main,
  },
  textUnselected: {
    fontWeight: "normal",
    color: theme.palette.common.black,
  },
  /**
   * Wrap link text in additional component to ensure
   * it's total width doesn't change when font weight
   * changes.
   */
  text: {
    ...theme.mixins.navlink,
    display: "inline-block",
    textAlign: "center",
    whiteSpace: "nowrap",
    /**
     * Reserve space for bolded content in ::before
     * pseudo element.
     */
    "&::before": {
      ...theme.mixins.navlink,
      fontWeight: "bold",
      display: "block",
      content: "attr(title)",
      height: 0,
      overflow: "hidden",
      visibility: "hidden",
    },
  },
}));

/**
 * Single navigation link.
 *
 * @see NavLinkList
 */
function NavLink(props) {
  const {
    link,
    selected,
    decorated = true,
    onClick,
    className,
    ...other
  } = props;
  const classes = useStyles();

  const decorator =
    selected && decorated ? <SelectionDecorator variant="bottom" /> : null;

  const handleClick = useCallback(() => {
    if (onClick != null) {
      onClick(link);
    }
  }, [link, onClick]);

  const labelId = useUniqueId("link-label");

  return (
    <ButtonBase
      className={clsx(classes.container, className)}
      onClick={handleClick}
      focusRipple
      disableTouchRipple
      aria-labelledby={labelId}
      component="div"
      {...other}
    >
      <div className={classes.link}>
        <div
          title={link.title}
          className={clsx(classes.text, {
            [classes.textSelected]: selected,
            [classes.textUnselected]: !selected,
          })}
          id={labelId}
        >
          {link.title}
        </div>
        {decorator}
      </div>
    </ButtonBase>
  );
}

export const LinkType = PropTypes.shape({
  title: PropTypes.string.isRequired,
});

NavLink.propTypes = {
  link: LinkType.isRequired,
  onClick: PropTypes.func,
  selected: PropTypes.bool,
  decorated: PropTypes.bool,
  className: PropTypes.string,
};

export default NavLink;
