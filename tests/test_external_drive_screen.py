import unittest

from acquisition.audit_external_drive_candidates import duplicate_screen


class ExternalDriveScreenTests(unittest.TestCase):
    def test_copies_with_new_ids_are_not_independent_candidates(self):
        development = [{'id': 'old', 'md5Checksum': 'hash', 'size': '123'}]
        candidates = [{'id': 'copy', 'md5Checksum': 'hash', 'size': '123'},
                      {'id': 'new', 'md5Checksum': 'different', 'size': '123'}]
        r = duplicate_screen(candidates, development)
        self.assertEqual(r['same_development_md5_size_count'], 1)
        self.assertFalse(r['candidates'][0]['eligible_for_provenance_review'])
        self.assertTrue(r['candidates'][1]['eligible_for_provenance_review'])

    def test_missing_hashes_and_internal_duplicates_are_explicit(self):
        r = duplicate_screen([{'id': 'a', 'md5Checksum': 'h', 'size': '1'},
                              {'id': 'b', 'md5Checksum': 'h', 'size': '1'}, {'id': 'c'}], [])
        self.assertEqual(r['candidate_hash_missing_count'], 1)
        self.assertEqual(r['candidate_duplicate_clusters'], [['a', 'b']])

    def test_existing_id_rejected_even_without_hash(self):
        r = duplicate_screen([{'id': 'old'}], [{'id': 'old'}])
        self.assertFalse(r['candidates'][0]['eligible_for_provenance_review'])
