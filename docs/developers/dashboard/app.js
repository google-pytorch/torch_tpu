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

// Interactive logic for the TorchTPU GitHub Work Dashboard.
//
// Security notes:
// - DOM content is created with createElement/textContent only; never use
//   innerHTML.
// - This static page has no backend, so the optional GitHub token is held
//   only in memory for the current tab and never stored or put in the URL.
(function() {
'use strict';

const config = window.DASHBOARD_CONFIG;
const CACHE_TTL_MS = 2 * 60 * 1000;  // Reuse search results for 2 minutes.
const PAGE_SIZE = 100;    // GitHub's maximum search results per request.
const MAX_RESULTS = 200;  // Each page costs one search against the limit.
const USERNAME_RE = /^[a-zA-Z0-9](?:[a-zA-Z0-9]|-(?=[a-zA-Z0-9])){0,38}$/;
const PRIORITIES = ['Urgent', 'High', 'Medium', 'Unset', 'Low'];

const $ = (id) => document.getElementById(id);
const roleSelect = $('role');
const usernameInput = $('username');
const extraInput = $('extra');

let view = config.views[0];
let sort = {col: 'updated', dir: 'asc'};
let token = '';
let items = [];
let fetchController = null;
const cache = new Map();  // "<query>|<has token>" -> fetched result.

// Creates an element with the given properties and children.
function el(tag, props, ...children) {
  const e = Object.assign(document.createElement(tag), props);
  e.append(...children);
  return e;
}

function link(href, text, className) {
  return el(
      'a',
      {href, textContent: text, className, target: '_blank', rel: 'noopener'});
}

// Only trust GitHub URLs from API responses.
function safeUrl(url) {
  return /^https:\/\/(github\.com|avatars\.githubusercontent\.com)\//.test(
             url || '') ?
      url :
      '';
}

function badge(text, className) {
  return el(
      'span', {className: 'badge ' + (className || ''), textContent: text});
}

function username() {
  const user = usernameInput.value.trim().replace(/^@/, '');
  return USERNAME_RE.test(user) ? user : '';
}

function withUser(query) {
  return username() ? query.replace(/@me\b/g, username()) : query;
}

function showBanner(message, level) {
  $('banner').hidden = !message;
  $('banner').textContent = message;
  $('banner').className = level || '';
}

// ----- Filters and presets -----

// Filters applicable to the current view, with the "All" option appended.
function viewFilters() {
  return config.filters
      .filter(
          (f) => (f.type === 'both' || f.type === view.type) &&
              (!f.roles || f.roles.includes(view.role)))
      .map((f) => ({
             ...f,
             options: f.all === false ?
                 f.options :
                 f.options.concat([['all', f.all || 'All', '']]),
           }));
}

function selectedFilters() {
  return Object.fromEntries([...$('filters').querySelectorAll('select')].map(
      (s) => [s.name, s.value]));
}

// Filter values for the view: `overrides`, then the filter's default, then
// "all" (skipping values that aren't options).
function filterValues(overrides) {
  return Object.fromEntries(
      viewFilters().map(
          (f) =>
              [f.id,
               [overrides[f.id], f.default, 'all'].find(
                   (v) => f.options.some(([option]) => option === v)),
  ]));
}

function setFilters(values) {
  for (const select of $('filters').querySelectorAll('select')) {
    select.value = values[select.name];
  }
  highlightPreset();
}

function highlightPreset() {
  const selected = JSON.stringify(selectedFilters());
  $('presets').querySelectorAll('button').forEach((button, i) => {
    button.classList.toggle(
        'active',
        JSON.stringify(filterValues(view.presets[i].filters)) === selected);
  });
}

// Renders the tabs, presets, and filters for `view`, applying `state`
// (filters, sort, and extra qualifiers, e.g. from the URL). By default the
// filters are set to the view's first preset.
function renderView(state) {
  $('tabs').replaceChildren(
      ...config.views.filter((v) => v.role === view.role)
          .map((v) => el('button', {
                 className: 'tab' + (v === view ? ' active' : ''),
                 textContent: v.label,
                 onclick: () => v !== view && selectView(v),
               })));
  $('ci-status').hidden = view.role !== 'oncaller';

  $('presets').replaceChildren(
      el('span', {className: 'label', textContent: 'Presets:'}),
      ...view.presets.map(
          (preset) => el('button', {
            className: 'btn',
            textContent: preset.label,
            onclick: () => (setFilters(filterValues(preset.filters)), update()),
          })));

  $('filters').replaceChildren(...viewFilters().map(
      (f) =>
          el('label', {}, f.label,
             el('select',
                {name: f.id, onchange: () => (highlightPreset(), update())},
                ...f.options.map(
                    ([value, label]) =>
                        el('option', {value, textContent: label}))))));
  setFilters(filterValues(state.filters || view.presets[0].filters));

  const saved = parseSort(state.sort || view.sort);
  sort = columnIds().includes(saved.col) ? saved : parseSort(view.sort);
  extraInput.value = state.extra || '';
}

function parseSort(value) {
  const [col, dir] = value.split('-');
  return {col, dir: dir || 'asc'};
}

function selectView(v) {
  view = v;
  roleSelect.value = v.role;
  renderView({});
  update();
}

// Saves the state to the URL (for bookmarking).
function saveHash() {
  const params =
      new URLSearchParams({view: view.id, sort: sort.col + '-' + sort.dir});
  for (const [id, value] of Object.entries(selectedFilters())) {
    params.set('f_' + id, value);
  }
  if (extraInput.value.trim()) {
    params.set('q', extraInput.value.trim());
  }
  history.replaceState(null, '', '#' + params);
}

function update() {
  saveHash();
  search();
}

function restoreFromHash() {
  const params = new URLSearchParams(location.hash.slice(1));
  const v = config.views.find((v) => v.id === params.get('view'));
  const filters = {};
  params.forEach((value, key) => {
    if (key.startsWith('f_')) {
      filters[key.slice(2)] = value;
    }
  });
  view = v || config.views[0];
  roleSelect.value = view.role;
  renderView(
      v ? {filters, sort: params.get('sort'), extra: params.get('q')} : {});
}

// ----- Search -----

function buildQuery() {
  const selected = selectedFilters();
  const parts = ['is:' + view.type];
  for (const f of viewFilters()) {
    const option = f.options.find(([value]) => value === selected[f.id]);
    if (option && option[2]) {
      parts.push(withUser(option[2]));
    }
  }
  if (extraInput.value.trim()) {
    parts.push(extraInput.value.trim());
  }
  // Results are sorted in the browser; this only decides which items load
  // first when there are more matches than MAX_RESULTS.
  parts.push(
      'sort:updated-' +
      (sort.col === 'updated' && sort.dir === 'desc' ? 'desc' : 'asc'));
  return parts.join(' ');
}

async function search(force) {
  const query = buildQuery();
  const apiQuery = 'repo:' + config.repo + ' ' + query;
  $('description').textContent = view.description;
  $('query').textContent = apiQuery;
  $('open').href = 'https://github.com/' + config.repo + '/' +
      (view.type === 'pr' ? 'pulls' : 'issues') +
      '?q=' + encodeURIComponent(query);
  showBanner('');

  if (view.role === 'ic' && !username()) {
    items = [];
    $('count').textContent = '0 items (GitHub username required)';
    renderTable();
    showBanner(
        usernameInput.value.trim() ?
            'Please enter a valid GitHub username (alphanumeric and single hyphens) in the top bar.' :
            'Enter your GitHub @username in the top bar to load "' +
                view.label +
                '", or click "Open on GitHub" to use @me directly on github.com.',
        usernameInput.value.trim() ? 'warning' : '');
    return;
  }

  const key = apiQuery + '|' + Boolean(token);
  const cached = cache.get(key);
  if (!force && cached && Date.now() - cached.fetchedAt < CACHE_TTL_MS) {
    showResults(cached);
    return;
  }

  $('count').textContent = 'Loading...';
  if (fetchController) {
    fetchController.abort();
  }
  fetchController = new AbortController();
  const headers = {Accept: 'application/vnd.github+json'};
  if (token) {
    headers.Authorization = 'Bearer ' + token;
  }

  const result = {items: [], totalCount: 0, fetchedAt: Date.now(), error: 0};
  try {
    for (let page = 1;; page++) {
      const response = await fetch(
          'https://api.github.com/search/issues?advanced_search=true&per_page=' +
              PAGE_SIZE + '&page=' + page +
              '&q=' + encodeURIComponent(apiQuery),
          {headers, signal: fetchController.signal});
      const remaining = response.headers.get('x-ratelimit-remaining');
      if (remaining !== null) {
        $('rate').textContent = 'Searches left this minute: ' + remaining +
            '/' + response.headers.get('x-ratelimit-limit') +
            (token ? ' (using your token)' : ' (anonymous)');
      }
      if (!response.ok) {
        result.error = response.status;
        break;
      }
      const data = await response.json();
      result.totalCount = data.total_count;
      result.items.push(...data.items);
      // Fetch more pages only while GitHub reports more matches.
      if (data.items.length < PAGE_SIZE ||
          result.items.length >= Math.min(result.totalCount, MAX_RESULTS)) {
        break;
      }
    }
  } catch (err) {
    if (err.name !== 'AbortError') {
      showBanner(
          'Network error while contacting api.github.com. Use "Open on GitHub" if offline or blocked.',
          'error');
      $('count').textContent = 'Network error';
    }
    return;
  }
  if (result.error && !result.items.length) {
    showFetchError(result.error);
    $('count').textContent = 'Unable to load results';
    return;
  }
  if (!result.error) {
    cache.set(key, result);  // Don't cache partial results.
  }
  showResults(result);
}

function showFetchError(status) {
  if (status === 401) {
    showBanner(
        'GitHub rejected your token (401). It may be expired or mistyped. ' +
            'Clear it or paste a new one in "GitHub token (optional)".',
        'error');
  } else if (status === 403 || status === 429) {
    showBanner(
        token ?
            'GitHub search rate limit reached (30 searches/min with a token). ' +
                'Wait about a minute and click Refresh, or use "Open on GitHub".' :
            'GitHub search rate limit reached (10 searches/min without a token, ' +
                'possibly shared with others on your network). Add a personal ' +
                'token in "GitHub token (optional)" to raise it to 30/min, wait ' +
                'about a minute and click Refresh, or use "Open on GitHub".',
        'warning');
  } else {
    showBanner(
        'GitHub API request failed with status ' + status + '.', 'error');
  }
}

function showResults(result) {
  // GitHub may ignore field.priority:Low (e.g. without a token), so also
  // apply the Priority filter here.
  const priority = selectedFilters().priority;
  items = result.items.filter(
      (item) => priority === undefined || priority === 'all' ||
          (getPriority(item) === 'Low') === (priority === 'low'));
  const filteredOut = result.items.length - items.length;
  const truncated = result.totalCount > result.items.length;
  const total = result.totalCount - filteredOut;
  const age = Math.floor((Date.now() - result.fetchedAt) / 1000);
  $('count').textContent = (truncated && filteredOut ? '~' : '') + total +
      (total === 1 ? ' item' : ' items') +
      (truncated ? ' (showing ' + items.length + ')' : '') +
      (age >= 5 ? ' \u00B7 loaded ' + age + 's ago' : '');

  if (result.error) {
    showFetchError(result.error);
  } else if (truncated) {
    showBanner(
        'Showing ' + result.items.length + ' of ' + result.totalCount +
        ' matching items. ' +
        (result.items.length >= MAX_RESULTS ?
             'The dashboard loads at most ' + MAX_RESULTS +
                 ' results per search, because every 100 results uses up one of GitHub\'s limited searches per minute. ' :
             'GitHub returned fewer results than it reported. ') +
        'To see the rest, narrow the filters (for example, set Status to Open or add an extra ' +
        'qualifier such as label:bug), or click "Open on GitHub" to browse all results on github.com.');
  }
  renderTable();
}

// ----- Results table -----

function getPriority(item) {
  const field = (item.issue_field_values ||
                 []).find((f) => f.issue_field_name === 'Priority');
  return (field && field.single_select_option &&
          field.single_select_option.name) ||
      'Unset';
}

const labels = (item) => (item.labels || []).map((l) => l.name);
const turnLabels = (item) =>
    labels(item).filter((name) => name.startsWith('stat:'));
const login = (user) => (user && user.login) || '';

function userChip(user) {
  const chip = link(safeUrl(user.html_url), '', 'user');
  chip.append(el('img', {src: safeUrl(user.avatar_url), alt: ''}), user.login);
  return chip;
}

function involvement(item) {
  const me = username().toLowerCase();
  const isAuthor = login(item.user).toLowerCase() === me;
  if (view.type === 'pr') {
    // PR results only match "seeking my review" or "authored by me".
    return [isAuthor ? 'Author' : 'Reviewer'];
  }
  const isAssignee =
      (item.assignees || []).some((a) => login(a).toLowerCase() === me);
  return [isAssignee && 'Assignee', isAuthor && 'Filer'].filter(Boolean);
}

function relativeTime(iso) {
  const minutes = Math.floor((Date.now() - Date.parse(iso)) / 60000);
  if (minutes < 1) return 'just now';
  if (minutes < 60) return minutes + 'm ago';
  if (minutes < 1440) return Math.floor(minutes / 60) + 'h ago';
  if (minutes < 43200) return Math.floor(minutes / 1440) + 'd ago';
  return iso.slice(0, 10);
}

// A column showing one of `values` as a badge, sorted in `values` order.
function stateColumn(label, valueOf, values) {
  return {
    label,
    key: (item) => values.indexOf(valueOf(item)),
    cell: (item) => badge(
        valueOf(item),
        {Open: 'blue',
         Draft: 'gray',
         Stale: 'gray',
         Closed: 'gray'}[valueOf(item)]),
  };
}

// Columns named after a filter id show that filter's value. Every column is
// sortable by `key`; `cell` returns the cell content.
const COLUMNS = {
  number: {
    label: '#',
    key: (item) => item.number,
    cell: (item) => link(safeUrl(item.html_url), '#' + item.number, 'num'),
  },
  title: {
    label: 'Title & Labels',
    key: (item) => item.title.toLowerCase(),
    cell: (item) =>
        [link(safeUrl(item.html_url), item.title, 'title'),
         // Turn and stale labels have their own columns.
         ...labels(item)
             .filter((name) => !name.startsWith('stat:') && name !== 'stale')
             .map((name) => badge(name)),
],
  },
  assignment: {
    label: 'Assignees',
    key: (item) => (item.assignees || []).map(login).join(),
    cell: (item) => (item.assignees || []).map(userChip),
  },
  involvement: {
    label: 'My Involvement',
    key: (item) => involvement(item).join(),
    cell: (item) => involvement(item).map((role) => badge(role)),
  },
  turn: {
    label: 'Turn',
    key: (item) => turnLabels(item).join(),
    cell: (item) =>
        turnLabels(item).map((name) => badge(name.slice(5), 'blue')),
  },
  priority: {
    label: 'Priority',
    key: (item) => PRIORITIES.indexOf(getPriority(item)),
    cell: (item) =>
        (getPriority(item) === 'Unset' ?
             [] :
             badge(getPriority(item), getPriority(item).toLowerCase())),
  },
  readiness: stateColumn(
      'Readiness', (item) => (item.draft ? 'Draft' : 'Ready'),
      ['Ready', 'Draft']),
  freshness: stateColumn(
      'Freshness',
      (item) => (labels(item).includes('stale') ? 'Stale' : 'Fresh'),
      ['Fresh', 'Stale']),
  status: stateColumn(
      'Status',
      (item) =>
          (item.state === 'open'                                ? 'Open' :
               item.pull_request && item.pull_request.merged_at ? 'Merged' :
                                                                  'Closed'),
      ['Open', 'Merged', 'Closed']),
  author: {
    label: 'Author',
    key: (item) => login(item.user).toLowerCase(),
    cell: (item) => userChip(item.user),
  },
  updated: {
    label: 'Updated',
    key: (item) => Date.parse(item.updated_at),
    cell: (item) => el(
        'span',
        {title: item.updated_at, textContent: relativeTime(item.updated_at)}),
  },
  comments: {
    label: 'Comments',
    key: (item) => item.comments,
    cell: (item) => String(item.comments),
  },
};

// # and Title, then one column per filter (in filter order), then the rest.
function columnIds() {
  const ids = ['number', 'title', ...viewFilters().map((f) => f.id)];
  if (!ids.includes('author')) ids.push('author');
  if (!ids.includes('assignment')) ids.push('assignment');
  return ids.concat(['updated', 'comments']);
}

function renderTable() {
  const ids = columnIds();
  $('thead').replaceChildren(el(
      'tr', {},
      ...ids.map(
          (id) =>
              el('th', {
                className: id,
                title: 'Sort by ' + COLUMNS[id].label,
                onclick: () => {
                  // Clicking the sorted column reverses the order. Sorting is
                  // done in the browser, so this doesn't search again.
                  sort = {
                    col: id,
                    dir: sort.col === id && sort.dir === 'asc' ? 'desc' : 'asc'
                  };
                  saveHash();
                  renderTable();
                },
              },
                 COLUMNS[id].label,
                 sort.col === id ? el('span', {
                   className: 'arrow',
                   textContent: sort.dir === 'asc' ? ' \u25B2' : ' \u25BC'
                 }) :
                                   ''))));

  const key = COLUMNS[sort.col].key;
  const sign = sort.dir === 'desc' ? -1 : 1;
  const cmp = (a, b) =>
      (typeof a === 'number' ? a - b : String(a).localeCompare(b));
  const sorted =
      items
          .map((item) => ({item, key: key(item)}))
          // Ties: least recently updated first.
          .sort(
              (a, b) => sign * cmp(a.key, b.key) ||
                  Date.parse(a.item.updated_at) - Date.parse(b.item.updated_at))
          .map((entry) => entry.item);

  $('tbody').replaceChildren(
      ...(sorted.length ?
              sorted.map((item) => el('tr', {}, ...ids.map((id) => {
                           const content = [].concat(COLUMNS[id].cell(item));
                           return el(
                               'td', {className: id},
                               ...(content.length ? content : ['\u2014']));
                         }))) :
              [el('tr', {}, el('td', {
                    colSpan: ids.length,
                    className: 'muted',
                    textContent: 'No matching items found.'
                  }))]));
}

// ----- Initialization -----

usernameInput.value = localStorage.getItem('tt_dashboard_username') || '';
restoreFromHash();
search();

roleSelect.onchange = () => selectView(config.views.find(
    (v) => v.role === roleSelect.value && v.type === view.type));
usernameInput.onchange = () => {
  localStorage.setItem(
      'tt_dashboard_username', usernameInput.value.trim().replace(/^@/, ''));
  search();
};
$('apply').onclick = () => update();
extraInput.onkeydown = (e) => e.key === 'Enter' && update();
$('refresh').onclick = () => search(true);
$('reset').onclick = () => selectView(view);
$('copy').onclick = () =>
    navigator.clipboard.writeText(buildQuery()).then(() => {
      $('copy').textContent = 'Copied!';
      setTimeout(() => ($('copy').textContent = 'Copy Query'), 1500);
    });

const setToken = (value) => {
  token = value;
  $('token').value = '';
  $('token-status').textContent = token ?
      'Token set for this tab: up to 30 searches/min.' :
      'No token: up to 10 searches/min (anonymous).';
};
setToken('');
$('token-use').onclick = () => (setToken($('token').value.trim()), search());
$('token').onkeydown = (e) => e.key === 'Enter' && $('token-use').click();
$('token-clear').onclick = () => (setToken(''), search());

window.addEventListener('hashchange', () => {
  restoreFromHash();
  search();
});
})();
