import React, { useCallback, useEffect, useMemo, useState } from "react";
import clsx from "clsx";
import PropTypes from "prop-types";
import { makeStyles } from "@material-ui/styles";
import { useIntl } from "react-intl";
import LoadingHeader from "../LoadingHeader";
import FileDetails from "../FileDetails";
import FileMatchHeader from "./FileMatchHeader";
import MatchSelector from "./MatchSelector";
import useFileMatches from "../../../../application/api/matches/useFileMatches";
import MatchAPI from "../../../../application/api/matches/MatchAPI";
import MatchOptions, { DefaultMatchOptions } from "./MatchOptions";
import { Collapse, Tooltip } from "@material-ui/core";
import SettingsIcon from "@material-ui/icons/Settings";
import IconButton from "@material-ui/core/IconButton";

const useStyles = makeStyles((theme) => ({
  root: {},
  header: {
    height: theme.spacing(10),
    padding: theme.spacing(2),
    display: "flex",
    alignItems: "center",
  },
  title: {
    ...theme.mixins.title3,
    fontWeight: "bold",
    flexGrow: 1,
  },
  loading: {
    margin: theme.spacing(2),
  },
  fileHeader: {
    marginTop: 0,
    margin: theme.spacing(2),
  },
  errorMessage: {
    minHeight: 150,
    ...theme.mixins.title2,
    color: theme.palette.action.textInactive,
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
  },
  options: {
    marginTop: 0,
    margin: theme.spacing(2),
  },
  optionsButton: {
    marginLeft: theme.spacing(1),
  },
}));

/**
 * Compare two matches.
 */
function matchComparator(first, second) {
  if (first.distance < second.distance) {
    return -1;
  } else if (first.distance > second.distance) {
    return 1;
  } else {
    return String(first.file.filename).localeCompare(second.file.filename);
  }
}

/**
 * Get i18n text.
 */
function useMessages() {
  const intl = useIntl();
  return {
    title: intl.formatMessage({ id: "file.match" }),
    loadError: intl.formatMessage({ id: "match.load.error" }),
    notMatch: intl.formatMessage({ id: "match.notMatch" }),
    noMatches: intl.formatMessage({ id: "match.noMatches" }),
    showOptions: intl.formatMessage({ id: "actions.showOptions" }),
  };
}

function MatchFiles(props) {
  const { motherFileId, matchFileId, onMatchFileChange, className, ...other } =
    props;
  const classes = useStyles();
  const messages = useMessages();
  const [options, setOptions] = useState(DefaultMatchOptions);
  const [showOptions, setShowOptions] = useState(false);

  const {
    matches: loadedMatches,
    error: matchError,
    resumeLoading: loadMatches,
    hasMore,
    progress,
  } = useFileMatches({
    fileId: motherFileId,
    fields: ["meta", "exif", "scenes"],
    filters: {
      falsePositive: null,
    },
  });

  const matches = useMemo(
    () =>
      loadedMatches
        .sort(matchComparator)
        .filter((match) => options.showFalsePositive || !match.falsePositive),
    [loadedMatches, options.showFalsePositive]
  );

  // Move to the first element when matches are loaded
  useEffect(() => {
    if (!hasMore && matches.length > 0 && matchFileId == null) {
      onMatchFileChange(matches[0].file.id);
    }
  }, [hasMore, onMatchFileChange, motherFileId]);

  // Get index of the selected match file
  const selected = matches.findIndex((match) => match.file.id === matchFileId);

  const handleSelectionChange = useCallback(
    (index) => {
      onMatchFileChange(matches[index].file.id);
    },
    [hasMore, onMatchFileChange, motherFileId, matches]
  );

  const matchAPI = MatchAPI.use();
  const handleDismiss = useCallback(
    async (match) => {
      try {
        // Change displayed match if needed
        if (!options.showFalsePositive) {
          if (selected + 1 < matches.length) {
            onMatchFileChange(matches[selected + 1].file.id);
          } else if (selected - 1 >= 0) {
            onMatchFileChange(matches[selected - 1].file.id);
          }
        }

        // Dismiss current match
        await matchAPI.deleteMatch(match);
      } catch (error) {
        console.error("Error deleting match", error, { error, match });
      }
    },
    [selected, matches, onMatchFileChange, options]
  );

  const handleRestore = useCallback(async (match) => {
    try {
      await matchAPI.restoreMatch(match);
    } catch (error) {
      console.error("Error restoring match", error, { error, match });
    }
  });

  const handleToggleOptions = useCallback(
    () => setShowOptions(!showOptions),
    [showOptions]
  );

  useEffect(() => {
    // Change displayed match if needed
    if (
      matches[selected] == null ||
      (!options.showFalsePositive && matches[selected]?.falsePositive)
    ) {
      if (selected + 1 < matches.length) {
        onMatchFileChange(matches[selected + 1].file.id);
      } else if (selected - 1 >= 0) {
        onMatchFileChange(matches[selected - 1].file.id);
      } else if (matches[selected] == null && matches.length > 0) {
        onMatchFileChange(matches[0].file.id);
      }
    }
  }, [options.showFalsePositive]);

  let content;
  if (hasMore) {
    content = (
      <LoadingHeader
        onRetry={loadMatches}
        errorMessage={messages.loadError}
        error={matchError}
        className={classes.loading}
        progress={progress}
      />
    );
  } else if (matches.length > 0 && selected >= 0) {
    content = (
      <div>
        <FileMatchHeader
          onDismiss={handleDismiss}
          onRestore={handleRestore}
          match={matches[selected]}
          className={classes.fileHeader}
          data-selector="MatchHeader"
        />
        <FileDetails file={matches[selected].file} />
      </div>
    );
  } else {
    const errorMessage =
      matches.length === 0 ? messages.noMatches : messages.notMatch;
    content = <div className={classes.errorMessage}>{errorMessage}</div>;
  }

  return (
    <div
      className={clsx(classes.root, className)}
      data-selector="MatchFiles"
      {...other}
    >
      <div className={classes.header}>
        <div className={classes.title}>
          {messages.title}
          <Tooltip title={messages.showOptions}>
            <IconButton
              size="small"
              variant="outlined"
              color="secondary"
              aria-label={messages.showOptions}
              onClick={handleToggleOptions}
              className={classes.optionsButton}
            >
              <SettingsIcon fontSize="small" />
            </IconButton>
          </Tooltip>
        </div>

        {!hasMore && (
          <MatchSelector
            matches={matches}
            selected={selected}
            onChange={handleSelectionChange}
          />
        )}
      </div>
      <Collapse in={showOptions}>
        <MatchOptions
          options={options}
          onChange={setOptions}
          className={classes.options}
        />
      </Collapse>
      {content}
    </div>
  );
}

MatchFiles.propTypes = {
  /**
   * Mother file id.
   */
  motherFileId: PropTypes.number.isRequired,
  /**
   * Match file id.
   */
  matchFileId: PropTypes.number,
  /**
   * Handle match file change.
   */
  onMatchFileChange: PropTypes.func.isRequired,
  className: PropTypes.string,
};

export default MatchFiles;
