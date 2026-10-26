import unittest

from higgsfield.internal.util import (
    convert_https_to_ssh,
    match_https_link,
    parse_origin_link_or_else,
)


class OriginLinkTests(unittest.TestCase):
    def test_public_github_https_url(self):
        link = "https://github.com/org/repo.git"

        self.assertTrue(match_https_link(link))
        self.assertEqual(convert_https_to_ssh(link), "git@github.com:org/repo.git")
        self.assertEqual(parse_origin_link_or_else(link), "git@github.com:org/repo.git")

    def test_github_enterprise_https_url(self):
        link = "https://github.company.com/org/repo.git"

        self.assertTrue(match_https_link(link))
        self.assertEqual(
            convert_https_to_ssh(link), "git@github.company.com:org/repo.git"
        )
        self.assertEqual(
            parse_origin_link_or_else(link), "git@github.company.com:org/repo.git"
        )

    def test_public_github_ssh_url(self):
        link = "git@github.com:org/repo.git"

        self.assertFalse(match_https_link(link))
        self.assertEqual(parse_origin_link_or_else(link), link)

    def test_github_enterprise_ssh_url(self):
        link = "git@github.company.com:org/repo.git"

        self.assertFalse(match_https_link(link))
        self.assertEqual(parse_origin_link_or_else(link), link)

    def test_malformed_https_url_is_rejected(self):
        for link in (
            "https://github.com/org/repo",
            "https://github.com/org/repo.git/extra",
            "https://github.com:invalid/org/repo.git",
            "https://github.com/org/repo.git\n",
        ):
            with self.subTest(link=link):
                self.assertFalse(match_https_link(link))
                self.assertIsNone(parse_origin_link_or_else(link))

    def test_malformed_ssh_url_is_rejected(self):
        for link in (
            "git@github.com:org/repo",
            "git@github.com:org/repo.git/extra",
            "git@github.com:/org/repo.git",
            "git@github.com:org/repo.git\n",
        ):
            with self.subTest(link=link):
                self.assertIsNone(parse_origin_link_or_else(link))

    def test_non_github_url_is_rejected(self):
        for link in (
            "https://gitlab.com/org/repo.git",
            "git@gitlab.com:org/repo.git",
            "https://bitbucket.org/org/repo.git",
            "git@bitbucket.org:org/repo.git",
        ):
            with self.subTest(link=link):
                self.assertFalse(match_https_link(link))
                self.assertIsNone(parse_origin_link_or_else(link))


if __name__ == "__main__":
    unittest.main()
