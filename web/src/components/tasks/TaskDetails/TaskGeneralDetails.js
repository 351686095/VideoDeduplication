import React from "react";
import clsx from "clsx";
import PropTypes from "prop-types";
import { makeStyles } from "@material-ui/styles";
import TaskType from "../../../prop-types/TaskType";
import TableBody from "@material-ui/core/TableBody";
import Table from "@material-ui/core/Table";
import { useIntl } from "react-intl";
import { taskAttributes } from "./taskAttributes";
import TableRow from "@material-ui/core/TableRow";
import TableCell from "@material-ui/core/TableCell";

const useStyles = makeStyles((theme) => ({
  root: {},
  attrName: {
    ...theme.mixins.valueNormal,
    color: theme.palette.action.textInactive,
  },
  attrValue: {
    ...theme.mixins.valueNormal,
    ...theme.mixins.textEllipsis,
    maxWidth: 300,
  },
}));

function TaskGeneralDetails(props) {
  const { task, className, ...other } = props;
  const classes = useStyles();
  const intl = useIntl();

  const attributes = taskAttributes.map((attr) => ({
    title: intl.formatMessage({ id: attr.title }),
    value: attr.value(task, intl),
  }));

  return (
    <div className={clsx(classes.root, className)} {...other}>
      <Table>
        <TableBody>
          {attributes.map((attr) => (
            <TableRow key={attr.title}>
              <TableCell className={classes.attrName}>{attr.title}</TableCell>
              <TableCell className={classes.attrValue}>{attr.value}</TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </div>
  );
}

TaskGeneralDetails.propTypes = {
  /**
   * Task that will be described.
   */
  task: TaskType.isRequired,
  className: PropTypes.string,
};

export default TaskGeneralDetails;
