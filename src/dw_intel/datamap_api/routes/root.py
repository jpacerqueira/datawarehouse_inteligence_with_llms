from fastapi import APIRouter

router = APIRouter()


@router.get("/", tags=["root"])
async def root():
    """Root endpoint returning API status"""
    return {"message": "Cashflow DataMap Schema API"}
