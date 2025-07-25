from fastapi import APIRouter

from . import csv_routes, labelling_routes, rfc_routes, files_routes

router = APIRouter()
router.include_router(csv_routes.router, prefix="/convert", tags=["convert"])
router.include_router(labelling_routes.router, tags=["label"])
router.include_router(rfc_routes.router, prefix="/rfc", tags=["rfc"])
router.include_router(files_routes.router, tags=["files"])


