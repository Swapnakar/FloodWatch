import asyncio
from services.imd_service import get_imd_service
async def main():
    imd = get_imd_service()
    res = await imd.get_weather()
    print("IMD API Result:", res)
asyncio.run(main())
