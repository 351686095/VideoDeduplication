import lodash from "lodash";
import { useEffect, useMemo, useState } from "react";
import useValue from "../../hooks/useValue";
import { useServer } from "../../../server-api/context";

/**
 * Load video objects using bare server client.
 */
export default function useLoadObjects({ filters, fields }) {
  const server = useServer();
  const [total, setTotal] = useState(undefined);
  const [objects, setObjects] = useState([]);
  const templatesCache = useMemo(() => new Map());
  const filesCache = useMemo(() => new Map());
  const params = useValue({ filters, fields });

  // Reset loaded objects when params are changed
  useEffect(() => {
    setTotal(undefined);
    setObjects([]);
  }, [params]);

  useEffect(() => {
    // Copy current request params
    const requestParams = lodash.merge({}, params);

    // If all objects are loaded, then do nothing
    if (total != null && total <= objects.length) {
      return;
    }

    // Define a function to load the next objects slice
    const loadObjects = async () => {
      const response = await server.fetchTemplateMatches({
        offset: objects.length,
        filters,
        fields,
      });
      if (!response.success) {
        throw response.error;
      }
      const {
        templateMatches: newObjects,
        files: loadedFiles,
        templates: loadedTemplates,
        total: updatedTotal,
      } = response.data;

      // Update template cache
      for (const template of loadedTemplates) {
        templatesCache.set(template.id, template);
      }

      // Update files cache
      for (const file of loadedFiles) {
        filesCache.set(file.id, file);
      }

      // Wire templates and files into loaded objects
      const wiredObjects = newObjects.map((object) => ({
        ...object,
        file: filesCache.get(object.fileId),
        template: templatesCache.get(object.templateId),
      }));

      // Calculate updated objects
      const updatedObjects = [...objects, ...wiredObjects];

      return { updatedTotal, updatedObjects };
    };

    // Apply changes if not cancelled.
    let cancel = false;
    loadObjects().then(({ updatedTotal, updatedObjects }) => {
      if (!cancel && lodash.isEqual(requestParams, params)) {
        setTotal(updatedTotal);
        setObjects(updatedObjects);
      }
    });

    // Cancel operation on cleanup
    return () => {
      cancel = true;
    };
  }, [total, objects, params]);

  const progress = total == null ? 0 : (100 * objects.length) / total;
  const done = total != null && total <= objects.length;
  return { objects, progress, total, done };
}
