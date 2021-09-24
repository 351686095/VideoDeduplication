import React, { useCallback, useMemo } from "react";
import PropTypes from "prop-types";
import FileExtensionPicker from "./FileExtensionPicker";
import { useFilters } from "../../../application/api/files/useFilters";
import FilterList from "./FilterList";
import DateRangeFilter from "./DateRangeFilter";
import BoolFilter from "./BoolFilter";
import { useIntl } from "react-intl";
import objectDiff from "../../../lib/helpers/objectDiff";
import useFileExtensions from "../../../application/api/stats/useFileExtensions";
import {
  parseDateRange,
  stringifyDateRange,
} from "../../../lib/helpers/date-range";
import useFilesColl from "../../../application/api/files/useFilesColl";
import { DefaultFilters } from "../../../model/VideoFile";

/**
 * Get i18n text.
 */
function useMessages() {
  const intl = useIntl();
  return {
    date: intl.formatMessage({ id: "filter.creationDate" }),
    dateHelp: intl.formatMessage({ id: "filter.creationDate.help" }),
    audio: intl.formatMessage({ id: "filter.hasAudio" }),
    audioHelp: intl.formatMessage({ id: "filter.hasAudio.help" }),
    origin: intl.formatMessage({ id: "filter.origin" }),
    originHelp: intl.formatMessage({ id: "filter.origin.help" }),
    originRemote: intl.formatMessage({ id: "filter.origin.remote" }),
    originLocal: intl.formatMessage({ id: "filter.origin.local" }),
  };
}

/**
 * Get count of active filters.
 */
function useActiveFilters() {
  const filters = useFilesColl().params;
  const diff = objectDiff(filters, DefaultFilters);
  return diff.extensions + diff.date + diff.audio + diff.remote;
}

function MetadataFilters(props) {
  const { className, ...other } = props;
  const [filters, setFilters] = useFilters();
  const extensions = useFileExtensions();
  const messages = useMessages();
  const dateRange = useMemo(() => parseDateRange(filters.date), [filters.date]);

  const handleUpdateExtensions = useCallback(
    (extensions) => setFilters({ extensions }),
    [setFilters]
  );

  const handleDateChange = useCallback(
    (date) => setFilters({ date: stringifyDateRange(date) }),
    [setFilters]
  );

  const handleAudioChange = useCallback(
    (audio) => setFilters({ audio }),
    [setFilters]
  );

  const handleRemoteChange = useCallback(
    (remote) => setFilters({ remote }),
    [setFilters]
  );

  return (
    <FilterList className={className} {...other}>
      <FileExtensionPicker
        selected={filters.extensions}
        onChange={handleUpdateExtensions}
        extensions={extensions}
      />
      <DateRangeFilter
        title={messages.date}
        range={dateRange}
        onChange={handleDateChange}
        tooltip={messages.dateHelp}
      />
      <BoolFilter
        title={messages.audio}
        value={filters.audio}
        onChange={handleAudioChange}
        tooltip={messages.audioHelp}
      />
      <BoolFilter
        title={messages.origin}
        value={filters.remote}
        onChange={handleRemoteChange}
        tooltip={messages.originHelp}
        trueText={messages.originRemote}
        falseText={messages.originLocal}
      />
    </FilterList>
  );
}

/**
 * Hook to retrieve active filters count.
 */
MetadataFilters.useActiveFilters = useActiveFilters;

MetadataFilters.propTypes = {
  className: PropTypes.string,
};

export default MetadataFilters;
