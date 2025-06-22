from fastapi import APIRouter

from dw_intel.datamap_api.routes.analyse import router as analyse_router
from dw_intel.datamap_api.routes.column import router as column_router
from dw_intel.datamap_api.routes.root import router as root_router
from dw_intel.datamap_api.routes.schema import router as schema_router
from dw_intel.datamap_api.routes.similar import router as similar_router
from dw_intel.datamap_api.routes.table import router as table_router
from dw_intel.datamap_api.routes.initialize import router as initialize_router
from dw_intel.datamap_api.routes.sql_in_context import router as sql_in_context_router
from dw_intel.datamap_api.routes.execute_sql import router as execute_sql_router
from dw_intel.datamap_api.routes.generate_erd import router as generate_erd_router
from dw_intel.datamap_api.routes.sql_in_context_v2 import router as sql_in_context_v2_router

router = APIRouter()

router.include_router(
    root_router,
    tags=["root"],
)

router.include_router(
    analyse_router,
    tags=["analysis"],
)

router.include_router(
    column_router,
    tags=["columns"],
)

router.include_router(
    schema_router,
    tags=["schema"],
)

router.include_router(
    similar_router,
    tags=["similar"],
)

router.include_router(
    table_router,
    tags=["tables"],
)

router.include_router(
    initialize_router,
    tags=["root"],
)

router.include_router(
    sql_in_context_router,
    tags=["analysis"],
)

router.include_router(
    execute_sql_router,
    tags=["execute_sql"],
)

router.include_router(
    generate_erd_router,
    tags=["generate_erd"],
)

router.include_router(
    sql_in_context_v2_router,
    tags=["analysis"],
)
