from flask import jsonify, request, url_for

from db.schema import Scene
from .blueprint import api


@api.route("/scenes/")
def get_scenes():
    page = request.args.get("page", 1, type=int)
    pagination = Scene.query.paginate(page, per_page=10, error_out=False)
    scenes = pagination.items

    prev = None
    if pagination.has_prev:
        prev = url_for("api.get_scenes", page=page - 1)
    next = None
    if pagination.has_next:
        next = url_for("api.get_scenes", page=page + 1)
    return jsonify(
        {"posts": [scene.to_json() for scene in scenes], "prev": prev, "next": next, "count": pagination.total}
    )
