"""Exercise the production monitor with independent lifecycle and rc observations."""
import asyncio
import time
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock

import httpx
from lagent.serving.sandbox.providers.dbox_client import DBoxTransportError

from xtuner.v1.rl.agent_loop.sandbox_agent_loop.sandbox import DetachedShellEntry, EntryCapture


class MonitorTests(unittest.IsolatedAsyncioTestCase):
    def entry(self,**kwargs):
        entry=DetachedShellEntry('true',capture=EntryCapture(root='/tmp/capture'),timeout=2,
            poll_interval_sec=.2,pid_check_retry_interval_sec=.001,**kwargs)
        entry._read_int_capture_file=AsyncMock(return_value=None)
        entry._wait_int_capture_file=AsyncMock(return_value=0)
        return entry

    async def run_monitor(self,entry,client):
        return await entry._wait_with_pid_poll(client,SimpleNamespace(id='task'),
            SimpleNamespace(name='agent',rc_file='/tmp/rc'),123)

    async def test_more_than_three_transient_errors_with_running_sandbox_recover(self):
        entry=self.entry()
        client=SimpleNamespace(is_pid_running=AsyncMock(side_effect=[httpx.ReadTimeout('private-url')]*4+[False]),
            get_lifecycle_state=AsyncMock(return_value='ready'))
        outcome=await self.run_monitor(entry,client)
        self.assertTrue(outcome.ok)
        self.assertEqual(outcome.details['pid_check_errors_total'],4)

    async def test_terminal_sandbox_is_not_retried_or_misreported_as_exit(self):
        entry=self.entry()
        client=SimpleNamespace(is_pid_running=AsyncMock(side_effect=httpx.ReadTimeout('private-url')),
            get_lifecycle_state=AsyncMock(return_value='failed'))
        outcome=await self.run_monitor(entry,client)
        self.assertFalse(outcome.ok)
        self.assertEqual(outcome.reason,'sandbox_unavailable')
        self.assertEqual(outcome.details['sandbox_state'],'failed')
        self.assertNotIn('private-url',str(outcome))
        self.assertEqual(client.is_pid_running.await_count,1)

    async def test_rc_file_can_prove_completion_during_a_monitor_transport_error(self):
        entry=self.entry()
        entry._read_int_capture_file.return_value=0
        client=SimpleNamespace(is_pid_running=AsyncMock(side_effect=httpx.ReadTimeout('private-url')),
            get_lifecycle_state=AsyncMock(return_value='ready'))
        outcome=await self.run_monitor(entry,client)
        self.assertTrue(outcome.ok)
        self.assertTrue(outcome.details['completed_during_monitor_error'])

    async def test_unknown_lifecycle_failure_never_proves_sandbox_is_terminal(self):
        entry=self.entry()
        client=SimpleNamespace(is_pid_running=AsyncMock(side_effect=httpx.ReadTimeout('private-url')),
            get_lifecycle_state=AsyncMock(side_effect=RuntimeError('query failed')))
        outcome=await self.run_monitor(entry,client)
        self.assertEqual(outcome.reason,'pid_monitor_error')
        self.assertEqual(client.is_pid_running.await_count,3)

    async def test_healthy_lifecycle_does_not_allow_unbounded_monitor_retries(self):
        entry=self.entry(pid_check_recovery_timeout_sec=.01)
        client=SimpleNamespace(is_pid_running=AsyncMock(side_effect=httpx.ReadTimeout('private-url')),
            get_lifecycle_state=AsyncMock(return_value='ready'))
        outcome=await self.run_monitor(entry,client)
        self.assertEqual(outcome.reason,'pid_monitor_error')
        self.assertLess(client.is_pid_running.await_count,100)

    async def test_cancelled_monitor_does_not_turn_into_timeout_or_success(self):
        entry=self.entry()
        client=SimpleNamespace(is_pid_running=AsyncMock(side_effect=asyncio.CancelledError()))
        with self.assertRaises(asyncio.CancelledError): await self.run_monitor(entry,client)

    async def test_503_or_404_with_running_lifecycle_recovers_after_four_failures(self):
        for status in (503,404,200):
            with self.subTest(status=status):
                entry=self.entry()
                exc=DBoxTransportError('sanitized failure',status_code=status,phase='execd_response_stream')
                client=SimpleNamespace(is_pid_running=AsyncMock(side_effect=[exc]*4+[False]),
                    get_lifecycle_state=AsyncMock(return_value='ready'))
                outcome=await self.run_monitor(entry,client)
                self.assertTrue(outcome.ok)
                self.assertEqual(outcome.details['pid_check_errors_total'],4)
                observations=outcome.details['monitor_observations']
                self.assertEqual(len(observations),4)
                self.assertTrue(all(row['transport_status']==status for row in observations))

    async def test_404_with_deleted_lifecycle_stops_without_waiting_for_rc(self):
        entry=self.entry()
        exc=DBoxTransportError('not found',status_code=404)
        client=SimpleNamespace(is_pid_running=AsyncMock(side_effect=exc),
            get_lifecycle_state=AsyncMock(return_value='deleted'))
        outcome=await self.run_monitor(entry,client)
        self.assertEqual(outcome.reason,'sandbox_unavailable')
        self.assertEqual(outcome.details['transport_status'],404)
        self.assertEqual(client.is_pid_running.await_count,1)
        entry._read_int_capture_file.assert_not_awaited()

    async def test_persistent_transport_failure_at_entry_deadline_is_not_agent_timeout(self):
        entry=self.entry()
        entry.timeout=.01
        async def stalled(pid): await asyncio.sleep(1)
        client=SimpleNamespace(is_pid_running=stalled,get_lifecycle_state=AsyncMock(return_value='ready'))
        outcome=await self.run_monitor(entry,client)
        self.assertEqual(outcome.reason,'pid_monitor_error')
        self.assertTrue(outcome.details['deadline_reached'])

    async def test_healthy_live_agent_at_deadline_remains_a_runtime_timeout(self):
        entry=self.entry()
        entry.timeout=.01
        client=SimpleNamespace(is_pid_running=AsyncMock(return_value=True))
        outcome=await self.run_monitor(entry,client)
        self.assertEqual(outcome.reason,'timeout')

    async def test_exit_between_last_poll_and_deadline_uses_final_rc(self):
        # The process is alive at the only poll, then exits and writes rc before
        # the deadline; the next poll would come after the deadline.
        entry=self.entry()
        entry.timeout=.05
        entry.poll_interval_sec=10
        entry._read_int_capture_file.return_value=0
        client=SimpleNamespace(is_pid_running=AsyncMock(return_value=True))
        outcome=await self.run_monitor(entry,client)
        self.assertTrue(outcome.ok)
        self.assertEqual(outcome.result.return_code,0)
        self.assertTrue(outcome.details['timeout_with_rc'])
        self.assertEqual(client.is_pid_running.await_count,1)
        entry._read_int_capture_file.assert_awaited_once()

    async def test_deleting_lifecycle_is_terminal_and_never_confused_with_process_exit(self):
        entry=self.entry()
        client=SimpleNamespace(is_pid_running=AsyncMock(side_effect=httpx.ReadTimeout('private URL')),
            get_lifecycle_state=AsyncMock(return_value='deleting'))
        outcome=await self.run_monitor(entry,client)
        self.assertEqual(outcome.reason,'sandbox_unavailable')
        self.assertEqual(client.is_pid_running.await_count,1)

    async def test_diagnostics_remain_bounded_and_preserve_first_and_last_failure(self):
        entry=self.entry()
        client=SimpleNamespace(is_pid_running=AsyncMock(side_effect=[httpx.ReadTimeout('private URL')]*50+[False]),
            get_lifecycle_state=AsyncMock(return_value='ready'))
        outcome=await self.run_monitor(entry,client)
        observations=outcome.details['monitor_observations']
        self.assertTrue(outcome.ok)
        self.assertEqual(outcome.details['pid_check_errors_total'],50)
        self.assertEqual(len(observations),32)
        self.assertEqual((observations[0]['poll'],observations[-1]['poll']),(1,50))
        self.assertNotIn('private URL',str(observations))

    async def test_recovery_budget_caps_retry_sleep(self):
        entry=self.entry(pid_check_recovery_timeout_sec=.03)
        entry.pid_check_retry_interval_sec=.4
        client=SimpleNamespace(is_pid_running=AsyncMock(side_effect=httpx.ReadTimeout('private URL')),
            get_lifecycle_state=AsyncMock(return_value='ready'))
        started=time.monotonic()
        outcome=await self.run_monitor(entry,client)
        self.assertLess(time.monotonic()-started,.15)
        self.assertEqual(outcome.reason,'pid_monitor_error')
        self.assertTrue(outcome.details['recovery_deadline_reached'])
        self.assertFalse(outcome.details['deadline_reached'])

    async def test_recovery_budget_caps_lifecycle_lookup(self):
        entry=self.entry(pid_check_recovery_timeout_sec=.03)
        async def stuck_state(): await asyncio.sleep(1)
        client=SimpleNamespace(is_pid_running=AsyncMock(side_effect=httpx.ReadTimeout('private URL')),
            get_lifecycle_state=stuck_state)
        started=time.monotonic()
        outcome=await self.run_monitor(entry,client)
        self.assertLess(time.monotonic()-started,.15)
        self.assertEqual(outcome.reason,'pid_monitor_error')
        entry._read_int_capture_file.assert_not_awaited()

    async def test_recovery_budget_caps_rc_lookup_without_repeating_it_at_deadline(self):
        entry=self.entry(pid_check_recovery_timeout_sec=.03)
        async def stuck_rc(*args): await asyncio.sleep(1)
        entry._read_int_capture_file=AsyncMock(side_effect=stuck_rc)
        client=SimpleNamespace(is_pid_running=AsyncMock(side_effect=httpx.ReadTimeout('private URL')),
            get_lifecycle_state=AsyncMock(return_value='ready'))
        started=time.monotonic()
        outcome=await self.run_monitor(entry,client)
        self.assertLess(time.monotonic()-started,.15)
        self.assertEqual(outcome.reason,'pid_monitor_error')
        self.assertEqual(entry._read_int_capture_file.await_count,1)


if __name__=='__main__': unittest.main()
