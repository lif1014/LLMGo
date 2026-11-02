"""Shared evaluation service lifecycle without restarting an active peer game."""
import asyncio
from contextlib import asynccontextmanager


class EvaluationReplica:
    def __init__(self, factory, wait_ready):
        self.factory,self.wait_ready = factory,wait_ready
        self.service,self.active,self.dirty = None,0,False
        self.condition = asyncio.Condition()

    @asynccontextmanager
    async def lease(self, game, attempt):
        async with self.condition:
            while self.dirty and self.active:
                await self.condition.wait()
            if self.service is None:
                service = self.factory(game,attempt)
                try:
                    await asyncio.to_thread(service.start)
                    error = await asyncio.to_thread(self.wait_ready,[service],1800)
                    if error:raise RuntimeError(error)
                except BaseException:
                    await asyncio.to_thread(service.stop)
                    raise
                self.service,self.dirty = service,False
            self.active += 1
            service = self.service
        failed = False
        try:
            yield service
        except BaseException:
            failed = True
            raise
        finally:
            async with self.condition:
                self.active -= 1
                self.dirty |= failed
                if self.dirty and not self.active:
                    stale,self.service = self.service,None
                    if stale is not None:await asyncio.to_thread(stale.stop)
                self.condition.notify_all()

    async def close(self):
        if self.active:
            raise RuntimeError('Cannot close a replica with live evaluation games')
        if self.service is not None:
            stale,self.service = self.service,None
            await asyncio.to_thread(stale.stop)
