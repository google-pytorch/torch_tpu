// Copyright 2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Declarative configuration for the TorchTPU GitHub Work Dashboard. Edit this
// file to add or change views, filters, and presets; no build step is needed.
// In queries, @me is replaced with the GitHub username entered on the page.

// "Awaiting me" turn queries, shared by the Turn filter and the presets.
const ISSUE_AWAITING_ME = '((assignee:@me label:stat:awaiting-assignee) OR ' +
    '(author:@me label:stat:awaiting-requester))';
const PR_AWAITING_ME =
    '((review-requested:@me label:stat:awaiting-reviewers) OR ' +
    '(author:@me label:stat:awaiting-requester))';

window.DASHBOARD_CONFIG = {
  repo: 'google-pytorch/torch_tpu',

  // Filter drop-downs, shown (and shown as result columns) in this order.
  // A filter applies to 'issue', 'pr', or 'both' item types and optionally
  // only to some roles; filters sharing an id are variants of one filter.
  // Options are [value, label, query]. An "All" option (labelled `all`) that
  // adds nothing to the query is appended unless `all` is false. `default` is
  // the value used when a preset doesn't set the filter (otherwise "all").
  filters: [
    {
      id: 'assignment',
      label: 'Owner Assignment',
      type: 'issue',
      roles: ['oncaller'],
      default: 'unassigned',
      options: [
        ['assigned', 'Assigned', 'has:assignee'],
        ['unassigned', 'Unassigned', 'no:assignee'],
      ],
    },
    {
      id: 'assignment',
      label: 'Reviewers/Assignee Assignment',
      type: 'pr',
      roles: ['oncaller'],
      default: 'unassigned',
      options: [
        [
          'assigned',
          'Assigned',
          // GitHub doesn't support querying for whether a PR has reviewers.
          // We rely on the existence of any stat:awaiting-* label to indicate
          // an assigned PR, as such labels are first added when a reviewer is
          // assigned.
          '(has:assignee OR label:stat:awaiting-requester ' +
              'OR label:stat:awaiting-assignee OR label:stat:awaiting-reviewers)',
        ],
        [
          'unassigned',
          'Unassigned',
          'no:assignee -label:stat:awaiting-requester ' +
              '-label:stat:awaiting-assignee -label:stat:awaiting-reviewers',
        ],
      ],
    },
    {
      id: 'involvement',
      label: 'My Involvement',
      type: 'issue',
      roles: ['ic'],
      all: false,
      default: 'either',
      options: [
        ['assigned', 'Assigned to me', 'assignee:@me'],
        ['filed', 'Filed by me', 'author:@me'],
        [
          'either', 'Assigned to or filed by me', '(assignee:@me OR author:@me)'
        ],
      ],
    },
    {
      id: 'involvement',
      label: 'My Involvement',
      type: 'pr',
      roles: ['ic'],
      all: false,
      default: 'either',
      options: [
        ['reviewer', 'Seeking my review', 'review-requested:@me'],
        ['author', 'Authored by me', 'author:@me'],
        [
          'either',
          'Seeking my review or authored by me',
          '(review-requested:@me OR author:@me)',
        ],
      ],
    },
    {
      id: 'turn',
      label: 'Turn',
      type: 'issue',
      roles: ['oncaller'],
      all: 'All turn states',
      options: [
        ['assignee', 'Awaiting assignee', 'label:stat:awaiting-assignee'],
        ['requester', 'Awaiting requester', 'label:stat:awaiting-requester'],
      ],
    },
    {
      id: 'turn',
      label: 'Turn',
      type: 'pr',
      roles: ['oncaller'],
      all: 'All turn states',
      options: [
        ['reviewers', 'Awaiting reviewers', 'label:stat:awaiting-reviewers'],
        ['requester', 'Awaiting requester', 'label:stat:awaiting-requester'],
      ],
    },
    // "Awaiting me": the turn is mine as the assignee or the filer.
    // Issues I filed and assigned to myself are never "Awaiting others".
    {
      id: 'turn',
      label: 'Turn',
      type: 'issue',
      roles: ['ic'],
      all: 'All turn states',
      options: [
        ['me', 'Awaiting me', ISSUE_AWAITING_ME],
        [
          'others',
          'Awaiting others',
          '((assignee:@me -author:@me label:stat:awaiting-requester) OR ' +
              '(author:@me -assignee:@me label:stat:awaiting-assignee))',
        ],
      ],
    },
    // "Awaiting me": the turn is mine as a requested reviewer or the author.
    {
      id: 'turn',
      label: 'Turn',
      type: 'pr',
      roles: ['ic'],
      all: 'All turn states',
      options: [
        ['me', 'Awaiting me', PR_AWAITING_ME],
        [
          'others',
          'Awaiting others',
          '((review-requested:@me label:stat:awaiting-requester) OR ' +
              '(author:@me label:stat:awaiting-reviewers))',
        ],
      ],
    },
    {
      id: 'priority',
      label: 'Priority',
      type: 'issue',
      default: 'notLow',
      options: [
        ['notLow', 'Not low-priority', '-field.priority:Low'],
        ['low', 'Low-priority', 'field.priority:Low'],
      ],
    },
    {
      id: 'readiness',
      label: 'Readiness',
      type: 'pr',
      default: 'ready',
      options: [
        ['ready', 'Ready', 'draft:false'],
        ['draft', 'Draft', 'draft:true'],
      ],
    },
    {
      id: 'freshness',
      label: 'Freshness',
      type: 'both',
      default: 'fresh',
      options: [
        ['fresh', 'Fresh', '-label:stale'],
        ['stale', 'Stale', 'label:stale'],
      ],
    },
    {
      id: 'status',
      label: 'Status',
      type: 'both',
      default: 'open',
      options: [
        ['open', 'Open', 'state:open'],
        ['closed', 'Closed', 'state:closed'],
      ],
    },
    {
      id: 'author',
      label: 'Author',
      type: 'pr',
      default: 'human',
      options: [
        ['human', 'Human', '-author:app/copybara-service'],
        ['copybara', 'Copybara', 'author:app/copybara-service'],
      ],
    },
  ],

  // `sort` is '<column>-<asc|desc>' (ascending priority means high priority
  // first). `presets` are one-click filter settings for common tasks, given as
  // changes to the filter defaults; a view starts with its first preset.
  views: [
    {
      id: 'oncaller-issues',
      role: 'oncaller',
      type: 'issue',
      label: 'Issues',
      description:
          'Issues for triage and owner assignment, ordered with high-priority items first.',
      sort: 'priority-asc',
      presets: [
        {label: 'Assign issue owners', filters: {}},
        {
          label: 'Check issue response time',
          filters: {assignment: 'assigned', turn: 'assignee'},
        },
      ],
    },
    {
      id: 'oncaller-prs',
      role: 'oncaller',
      type: 'pr',
      label: 'PRs',
      description:
          'Pull requests ordered by oldest update first so long-waiting PRs are surfaced.',
      sort: 'updated-asc',
      presets: [
        {label: 'Assign PR reviewers', filters: {}},
        {
          label: 'Check PR response time',
          filters: {assignment: 'assigned', turn: 'reviewers'},
        },
      ],
    },
    {
      id: 'ic-issues',
      role: 'ic',
      type: 'issue',
      label: 'Issues',
      description:
          'Issues assigned to or filed by your GitHub username, ordered with high-priority items first.',
      sort: 'priority-asc',
      presets: [
        {
          label: 'My TODOs',
          filters: {
            involvement: 'assigned',
            turn: 'me',
            priority: 'all',
            freshness: 'all',
          },
        },
      ],
    },
    {
      id: 'ic-prs',
      role: 'ic',
      type: 'pr',
      label: 'PRs',
      description:
          'Pull requests seeking your review or authored by your GitHub username.',
      sort: 'updated-asc',
      presets: [
        {
          label: 'My TODOs',
          filters: {turn: 'me', readiness: 'all', freshness: 'all'},
        },
      ],
    },
  ],
};
