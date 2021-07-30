import React, { useCallback } from "react";
import clsx from "clsx";
import PropTypes from "prop-types";
import { makeStyles } from "@material-ui/styles";
import { ButtonBase } from "@material-ui/core";
import Badge from "@material-ui/core/Badge";

const useStyles = makeStyles((theme) => ({
  tab: {
    cursor: "pointer",
    borderBottom: `3px solid rgba(0,0,0,0)`,
    paddingBottom: theme.spacing(0.5),
    marginLeft: ({ indent }) => theme.spacing(indent),
    flexShrink: 0,
  },
  sizeLarge: {
    ...theme.mixins.navlinkLarge,
    fontWeight: 500,
  },
  sizeMedium: {
    ...theme.mixins.navlink,
    fontWeight: 500,
  },
  sizeSmall: {
    ...theme.mixins.navlinkSmall,
    fontWeight: 500,
  },
  inactive: {
    color: theme.palette.action.textInactive,
  },
  selected: {
    borderBottom: `3px solid ${theme.palette.primary.main}`,
  },
  disabled: {
    cursor: "not-allowed",
  },
}));

/**
 * Get tab's label CSS class
 */
function labelClass(classes, size, selected) {
  return clsx({
    [classes.sizeMedium]: size === "medium",
    [classes.sizeSmall]: size === "small",
    [classes.sizeLarge]: size === "large",
    [classes.inactive]: !selected,
  });
}

/**
 * Get spacing between tabs.
 */
function getIndent({ first, spacing }) {
  if (first) {
    return 0;
  }
  return 4 * spacing;
}

/**
 * Array of selectable tabs
 */
function SelectableTab(props) {
  const {
    label,
    selected,
    onSelect,
    value,
    size = "medium",
    className,
    badge,
    badgeMax,
    badgeColor = "default",
    disabled = false,
    first = true,
    spacing = 1,
    ...other
  } = props;
  const indent = getIndent({ first, spacing });
  const classes = useStyles({ indent });

  const handleSelect = useCallback(() => {
    if (!disabled) {
      onSelect(value);
    }
  }, [disabled, onSelect, value]);

  return (
    <ButtonBase
      onClick={handleSelect}
      className={clsx(
        classes.tab,
        selected && !disabled && classes.selected,
        disabled && classes.disabled,
        className
      )}
      component="div"
      focusRipple
      disableTouchRipple
      role="tab"
      aria-label={label}
      {...other}
    >
      <Badge badgeContent={badge} max={badgeMax} color={badgeColor}>
        <div className={labelClass(classes, size, selected, disabled)}>
          {label}
        </div>
      </Badge>
    </ButtonBase>
  );
}

SelectableTab.propTypes = {
  /**
   * Text that will be displayed as a tab label
   */
  label: PropTypes.string.isRequired,
  /**
   * Determine whether tab is selected
   */
  selected: PropTypes.bool,
  /**
   * Fires when tab is selected
   */
  onSelect: PropTypes.func,
  /**
   * Value identifying the tab
   */
  value: PropTypes.any,
  /**
   * Size variants
   */
  size: PropTypes.oneOf(["small", "medium", "large"]),
  /**
   * The value displayed with the optional badge.
   */
  badge: PropTypes.node,
  /**
   * The color of the component. It supports those theme colors that make sense
   * for this component.
   */
  badgeColor: PropTypes.oneOf(["default", "error", "primary", "secondary"]),
  /**
   * Max count to show in badge (if the value is numeric).
   */
  badgeMax: PropTypes.number,
  /**
   * Indicates that tab cannot be activated.
   */
  disabled: PropTypes.bool,
  /**
   * Indicates tab is the first (always set by  enclosing SelectableTabs
   * component). Required for auto-spacing between tabs.
   */
  first: PropTypes.bool,
  /**
   * Controls auto-spacing between tabs.
   */
  spacing: PropTypes.number,
  className: PropTypes.string,
};

export default SelectableTab;
