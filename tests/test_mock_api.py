from __future__ import annotations

import unittest
from fastapi import HTTPException
from mock_api.app import health, vendor_risk


class MockApiTests(unittest.TestCase):
    def test_health(self):
        self.assertEqual(health()['status'], 'ok')

    def test_known_vendor(self):
        self.assertEqual(vendor_risk('BrandBoard')['security_review_status'], 'not_completed')

    def test_forced_outage(self):
        with self.assertRaises(HTTPException) as caught:
            vendor_risk('NimbusAI')
        self.assertEqual(caught.exception.status_code, 503)


if __name__ == '__main__':
    unittest.main()
