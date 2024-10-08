/* = = = = = = = = =
AUTHOR: SINGLABHARAT
= = = = = = = = = */

/* = =
Arrays
= = */

vector<int> pref_sums(vector<int> &a) {
    int n = size(a);
    vector<int> ps = a;
    for (int i = 1; i < n; i++) ps[i] += ps[i - 1];
    return ps;
}

vector<vector<int>> pref_sums_2D(vector<vector<int>> &a) {
    int n = size(a), m = size(a[0]);
    vector<vector<int>> ps(n + 1, vector<int>(m + 1));
    for (int i = 1; i <= n; i++) {
        for (int j = 1; j <= m; j++) {
            ps[i][j] = a[i - 1][j - 1] + ps[i][j - 1] + ps[i - 1][j] - ps[i - 1][j - 1];
        }
    }
    return ps;
}

auto sum = [&](int r1, int c1, int r2, int c2) {
    r1++, c1++, r2++, c2++;
    return ps[r2][c2] - ps[r2][c1 - 1] - ps[r1 - 1][c2] + ps[r1 - 1][c1 - 1];
};

vector<int> suff_sums(vector<int> &a) {
    int n = size(a);
    vector<int> ss = a;
    for (int i = n - 2; i >= 0; i--) ss[i] += ss[i + 1];
    return s;
}

int max_subarr_sum(vector<int> &a) {
    int res = -1e9, curr = 0;
    for (int i : a) {
        curr = max(curr + i, i);
        res = max(res, curr);
    }
    return res;
}

int sliding_window(vector<int> &a, int k) {
    int n = size(a);
    int curr = accumulate(begin(a), begin(a) + k, 0ll);
    int res = curr;
    for (int l = 1, r = k; r < n; l++, r++) {
        curr = curr - a[l - 1] + a[r];
        res = max(res, curr);
    }
    return res;
}

vector<vector<int>> matrix_mul(vector<vector<int>> &a1, vector<vector<int>> &a2) {
    int n1 = size(a1), m1 = size(a1[0]), m2 = size(a2[0]);
    vector<vector<int>> res(n1, vector<int>(m2));
    for (int i = 0; i < n1; i++) {
        for (int j = 0; j < m2; j++) {
            for (int k = 0; k < m1; k++) res[i][j] += a1[i][k] * a2[k][j];
        }
    }
    return res;
}

/* = = =
Sorting
= = = */

void bubble_sort(vector<int> &a) {
    int n = size(a);
    for (int i = 0; i < n - 1; i++) {
        for (int j = 0; j < n - i - 1; j++) {
            if (a[j] > a[j + 1]) swap(a[j], a[j + 1]);
        }
    }
}

void insertion_sort(vector<int> &a) {
    int n = size(a);
    for (int j = 1; j < n; j++) {
        for (int i = j - 1; i >= 0 and a[i] > a[i + 1]; i--) {
            swap(a[i], a[i + 1]);
        }
    }
}

void selection_sort(vector<int> &a) {
    int n = size(a);
    for (int i = 0; i < n - 1; i++) {
        int mni = i;
        for (int j = i + 1; j < n; j++) {
            if (a[j] < a[mni]) mni = j;
        }
        swap(a[i], a[mni]);
    }
}

void merge(vector<int> &a, int l, int r) {
    int n = r - l + 1;
    int m = (l + r) / 2;
    vector<int> temp(n);
    int i = l, j = m + 1, k = 0;
    while (i <= m and j <= r) {
        if (a[i] < a[j]) temp[k++] = a[i++];
        else temp[k++] = a[j++];
    }
    while (i <= m) temp[k++] = a[i++];
    while (j <= r) temp[k++] = a[j++];
    for (int i = 0; i < n; i++) a[l + i] = temp[i];
}

void merge_sort(vector<int> &a, int l, int r) {
    if (l >= r) return;
    int m = (l + r) / 2;
    merge_sort(a, l, m);
    merge_sort(a, m + 1, r);
    merge(a, l, r);
}

int partition(vector<int> &a, int l, int r) {
    int pivot = a[r];
    int pi = l;
    for (int i = l; i < r; i++) {
        if (a[i] <= pivot) swap(a[i], a[pi++]);
    }
    swap(a[pi], a[r]);
    return pi;
}

void quick_sort(vector<int> &a, int l, int r) {
    if (l >= r) return;
    int pi = partition(a, l, r);
    quick_sort(a, l, pi - 1);
    quick_sort(a, pi + 1, r);
}

vector<int> counting_sort(vector<int> &a) {
    int mx = *max_element(begin(a), end(a));
    vector<int> freq(mx + 1);
    for (int i : a) freq[i]++;
    vector<int> res;
    for (int i = 0; i <= mx; i++) {
        while (freq[i]--) res.push_back(i);
    }
    a = res;
}

/* = = = =
Searching
= = = = */

int binary_search(int x, vector<int> &a) {
    l = 0, r = (int)size(a) - 1;
    while (l <= r) {
        int m = (l + r) / 2;
        if (a[m] == x) return m;
        if (a[m] > x) r = m - 1;
        else l = m + 1;
    }
    return -1;
}

auto ok = [&](int m) -> bool {};

auto binary_search = [&]() -> int {
    int res = -1, l = 0, r = 1e9;
    while (l <= r) {
        int m = (l + r) / 2;
        if (ok(m)) {
            res = m;
            r = m - 1;
        } else {
            l = m + 1;
        }
    }
    return res;
};

const double EPS = 1e-9;

auto f = [&](double x) -> double {};

auto ternary_search = [&]() -> double {
    double l = 0, r = 1e9;
    while (r - l > EPS) {
        double m1 = l + (r - l) / 3;
        double m2 = r - (r - l) / 3;
        if (f(m1) < f(m2)) l = m1;
        else r = m2;
    }
    return f(l);
};

/* = = = = = =
Number Theory
= = = = = = */

bool prime(int n) {
    if (n < 2) return 0;
    for (int i = 2; i * i <= n; i++) {
        if (n % i == 0) return 0;
    }
    return 1;
}

set<int> factors(int n) {
    set<int> f;
    for (int i = 1; i * i <= n; i++) {
        if (n % i == 0) f.insert(i), f.insert(n / i);
    }
    return f;
}

vector<pair<int, int>> prime_factors(int n) {
    vector<pair<int, int>> pf;
    for (int i = 2; i * i <= n; i++) {
        if (n % i == 0) {
            int p = 0;
            while (n % i == 0) {
                n /= i;
                p++;
            }
            pf.push_back({i, p});
        }
    }
    if (n > 1) pf.push_back({n, 1});
    return pf;
}

vector<bool> sieve_of_eratosthenes(int n) {
    vector<bool> prime(n, 1);
    prime[0] = prime[1] = 0;
    for (int i = 2; i * i < n; i++) {
        if (prime[i]) {
            for (int j = i * i; j < n; j += i) prime[j] = 0;
        }
    }
    return prime;
}

vector<int> spf_sieve(int n) {
    vector<int> spf(N, -1);
    for (int i = 2; i < N; i++) {
        if (spf[i] == -1) {
            for (int j = i; j < N; j += i) {
                if (spf[j] == -1) spf[j] = i;
            }
        }
    }
    return spf;
}

struct Mint {
    int val;

    Mint(int v = 0) {
        val = v % MOD;
        if (val < 0) val += MOD;
    }

    explicit operator int() {
        return val;
    }

    Mint& operator+=(const Mint &other) {
        val += other.val;
        if (val >= MOD) val -= MOD;
        return *this;
    }

    Mint& operator-=(const Mint &other) {
        val -= other.val;
        if (val < 0) val += MOD;
        return *this;
    }

    Mint& operator*=(const Mint &other) {
        val = (val * other.val) % MOD;
        return *this;
    }

    friend Mint power(Mint a, int b) {
        Mint res = 1;
        while (b) {
            if (b & 1) res *= a;
            a *= a;
            b >>= 1;
        }
        return res;
    }

    friend Mint inverse(Mint a) {
        return power(a, MOD - 2);
    }

    Mint &operator/=(const Mint &other) {
        return *this *= inverse(other);
    }

    friend Mint operator+(Mint a, const Mint other) {
        return a += other;
    }

    friend Mint operator-(Mint a, const Mint other) {
        return a -= other;
    }

    friend Mint operator-(const Mint a) {
        return 0 - a;
    }

    friend Mint operator*(Mint a, const Mint other) {
        return a *= other;
    }

    friend Mint operator/(Mint a, const Mint other) {
        return a /= other;
    }

    friend std::ostream &operator<<(std::ostream &os, Mint const& a) {
        return os << a.val;
    }

    friend std::istream &operator>>(std::istream &is, Mint &a) {
        is >> a.val;
        a.val %= MOD;
        if (a.val < 0) a.val += MOD;
        return is;
    }

    friend bool operator==(const Mint &a, const Mint &other) {
        return a.val == other.val;
    }

    friend bool operator!=(const Mint &a, const Mint &other) {
        return a.val != other.val;
    }
};

vector<Mint> fact;

Mint C(int n, int r) {
    if (r < 0 or r > n) return 0;
    return fact[n] / fact[r] / fact[n - r];
}

fact.resize(1e5);
fact[0] = 1;
for (int i = 1; i < MX; i++) fact[i] = fact[i - 1] * i;

int gcd(int a, int b) {
    return (b == 0 ? a : gcd(b, a % b));
}

pair<int, int> extended_gcd(int a, int b) {
    if (b == 0) return {1, 0};
    auto [x2, y2] = extended_gcd(b, a % b);
    return {y2, x2 - a / b * y2};
}


/* = = =
Strings
= = = */

Mint my_hash(string s) {
    int n = size(s);
    Mint h = 0;
    for (int i = 0; i < n; i++) h = h * 31 + (s[i] - 'a' + 1);
    return h;
}

vector<int> rabin_karp(string &s, string &t) {
    int n = size(s), m = size(t);
    Mint p = power(Mint(31), m - 1);
    Mint ht = my_hash(t);
    Mint hs = my_hash(s.substr(0, m));
    vector<int> pos;
    if (hs == ht) pos.push_back(0);
    for (int l = 1, r = m; r < n; l++, r++) {
        hs = (hs - (s[l - 1] - 'a' + 1) * p) * 31 + (s[r] - 'a' + 1);
        if (hs == ht) pos.push_back(l);
    }
    return pos;
}

vector<int> pref_func(string &s) {
    int n = size(s);
    vector<int> lps(n);
    for (int i = 1; i < n; i++) {
        int j = lps[i - 1];
        while (j > 0 and s[j] != s[i]) j = lps[j - 1];
        lps[i] = j + (s[j] == s[i]);
    }
    return lps;
}

vector<int> kmp(string &s, string &t) {
    int n = size(s), m = size(t);
    string temp = t + "#" + s;
    vector<int> lps = pref_func(temp);
    vector<int> pos;
    for (int i = m + 1; i < n + m + 1; i++) {
        if (lps[i] == m) pos.push_back(i - 2 * m);
    }
    return pos;
}

/* = = = = = = = = = = =
Dynamic Programming (DP)
= = = = = = = = = = = */

int knapsack(vector<pair<int, int>> &items, int cap) {
    vector<int> dp(cap + 1);
    for (auto [w, c] : items) {
        for (int i = cap; i >= w; i--) dp[i] = max(dp[i], c + dp[i - w]);
    }
    return dp[cap];
}

int longest_common_subseq(vector<int> &a, vector<int> &b) {
    int n = size(a), m = size(b);
    vector<vector<int>> dp(n + 1, vector<int>(m + 1));
    for (int i = 1; i <= n; i++) {
        for (int j = 1; j <= m; j++) {
            if (a[i - 1] == b[j - 1]) dp[i][j] = dp[i - 1][j - 1] + 1;
            else dp[i][j] = max(dp[i - 1][j], dp[i][j - 1]);
        }
    }
    return dp[n][m];
}

int longest_common_subarr(vector<int> &a, vector<int> &b) {
    int n = size(a), m = size(b);
    vector<vector<int>> dp(n + 1, vector<int>(m + 1));
    int res = 0;
    for (int i = 1; i <= n; i++) {
        for (int j = 1; j <= m; j++) {
            if (a[i - 1] == b[j - 1]) dp[i][j] = dp[i - 1][j - 1] + 1;
            res = max(res, dp[i][j]);
        }
    }
    return res;
}

int longest_inc_subseq(vector<int> &a) {
    vector<int> lis;
    for (int i : a) {
        auto it = lower_bound(begin(lis), end(lis), i);
        if (it == end(lis)) lis.push_back(i);
        else *it = i;
    }
    return size(lis);
}

int edit_distance(string &s, string &t) {
    int n = size(s), m = size(t);
    vector<vector<int>> dp(n + 1, vector<int>(m + 1));
    for (int i = 0; i <= n; i++) dp[i][0] = i;
    for (int j = 0; j <= m; j++) dp[0][j] = j;
    for (int i = 1; i <= n; i++) {
        for (int j = 1; j <= m; j++) {
            if (s[i - 1] == t[j - 1]) dp[i][j] = dp[i - 1][j - 1];
            else dp[i][j] = min({dp[i - 1][j - 1], dp[i - 1][j], dp[i][j - 1]}) + 1;
        }
    }
    return dp[n][m];
}

vector<int> digts(int n) {
    vector<int> d;
    while (n > 0) {
        d.push_back(n % 10);
        n /= 10;
    }
    reverse(begin(d), end(d));
    return d;
}

int dp[19][163][2];

int rec(int pos, int sod, bool tight, vector<int> &num) {
    if (pos == size(num)) return sod == ?;
    int &res = dp[pos][sod][tight];
    if (res != -1) return res;
    res = 0;
    int lim = (tight ? num[pos] : 9);
    for (int d = 0; d <= lim; d++) {
        int new_sod = sod + d;
        bool new_tight = tight and d == lim;
        res += rec(pos + 1, new_sod, new_tight, num);
    }
    return res;
}

int digit_dp(int l, int r) {
    memset(dp, -1, sizeof(dp));
    vector<int> dl = digts(l - 1);
    int ansl = (l == 0 ? 0 : rec(0, 0, true, dl));
    memset(dp, -1, sizeof(dp));
    vector<int> dr = digts(r);
    int ansr = rec(0, 0, true, dr);
    return ansr - ansl;
}


/* = =
Graphs
= = */

int n, m;
cin >> n >> m;
vector<vector<int>> g(n);
for (int i = 0; i < m; i++) {
    int u, v;
    cin >> u >> v;
    u--, v--;
    g[u].push_back(v);
    g[v].push_back(u);
}

int n, m;
cin >> n >> m;
vector<vector<pair<int, int>>> g(n);
for (int i = 0; i < m; i++) {
    int u, v, w;
    cin >> u >> v >> w;
    u--, v--;
    g[u].push_back(pair(v, w));
    g[v].push_back(pair(u, w));
}

vector<bool> vis(n);
function<void(int)> dfs = [&](int u) {
    vis[u] = true;
    for (int v : g[u]) {
        if (!vis[v]) {
            dfs(v);
        }
    }
};

function<void(int, int)> dfs = [&](int u, int p) {
    for (int v : tree[u]) {
        if (v != p) {
            dfs(v, u);
        }
    }
};

const vector<pair<int, int>> DIRS = {{ -1, 0}, {0, 1}, {1, 0}, {0, -1}};
vector<vector<bool>> vis(n, vector<bool>(m));

auto valid = [&](int i, int j) -> bool {
    return 0 <= i and i < n and 0 <= j and j < m and !vis[i][j];
};

function<void(int, int)> dfs = [&](int i, int j) {
    vis[i][j] = true;
    for (auto [di, dj] : DIRS) {
        int new_i = i + di, new_j = j + dj;
        if (valid(new_i, new_j)) {
            dfs(new_i, new_j);
        }
    }
};

auto bfs = [&](int src) -> vector<int> {
    vector<int> dist(n, INF);
    queue<int> q;
    q.push(src);
    dist[src] = 0;
    while (!empty(q)) {
        int u = q.front();
        q.pop();
        for (int v : g[u]) {
            if (dist[v] == INF) {
                dist[v] = dist[u] + 1;
                q.push(v);
            }
        }
    }
    return dist;
};

auto dijkstra = [&](int src) -> vector<int> {
    priority_queue<pair<int, int>, vector<pair<int, int>>, greater<pair<int, int>>> pq;
    vector<int> dist(n, INF);
    pq.push({0, src});
    dist[src] = 0;
    while (!empty(pq)) {
        auto [d, u] = pq.top();
        pq.pop();
        if (dist[u] != d) continue;
        for (auto [v, w] : g[u]) {
            if (dist[u] + w < dist[v]) {
                dist[v] = dist[u] + w;
                pq.push({dist[v], v});
            }
        }
    }
    return dist;
};

auto bellman_ford = [&](int src) -> vector<int> {
    vector<int> dist(n, 1e9);
    dist[src] = 0;
    for (int _ = 0; _ < n - 1; _++) {
        for (auto e : edges) {
            int u = e[0], v = e[1], w = e[2];
            if (dist[u] + w < dist[v]) dist[v] = dist[u] + w;
        }
    }
    return dist;
};

auto floyd_warshall = [&]() -> vector<vector<int>> {
    vector<vector<int>> dist = g;
    for (int k = 0; k < n; k++) {
        for (int u = 0; u < n; u++) {
            for (int v = 0; v < n; v++) {
                dist[u][v] = min(dist[u][v], dist[u][k] + dist[k][v]);
            }
        }
    }
    return dist;
};

auto topo_sort = [&]() -> vector<int> {
    vector<int> indeg(n);
    for (int u = 0; u < n; u++) {
        for (int v : g[u]) indeg[v]++;
    }
    vector<int> topo;
    queue<int> q;
    for (int i = 0; i < n; i++) if (indeg[i] == 0) q.push(i);
    while (!empty(q)) {
        int u = q.front();
        q.pop();
        topo.push_back(u);
        for (int v : g[u]) {
            indeg[v]--;
            if (indeg[v] == 0) q.push(v);
        }
    }
    return topo;
};

auto scc = [&]() -> vector<vector<int>> {
    vector<int> topo;
    vector<bool> vis(n);

    function<void(int)> dfs = [&](int u) {
        vis[u] = true;
        for (int v : g[u]) {
            if (!vis[v]) dfs(v);
        }
        topo.push_back(u);
    };

    for (int u = 0; u < n; u++) {
        if (!vis[u]) dfs(u);
    }

    reverse(begin(topo), end(topo));

    vector<vector<int>> revg(n);
    for (int u = 0; u < n; u++) {
        for (int v : g[u]) revg[v].push_back(u);
    }

    vector<int> who(n, -1);
    function<void(int, int)> revdfs = [&](int u, int p) {
        who[u] = p;
        for (int v : revg[u]) {
            if (who[v] == -1) revdfs(v, p);
        }
    };

    for (int u : topo) {
        if (who[u] == -1) dfsrev(u, u);
    }

    vector<vector<int>> compg(n);
    for (int u = 0; u < n; u++) {
        for (int v : g[u]) {
            if (who[u] != who[v]) compg[who[u]].push_back(who[v]);
        }
    }

    return compg;
};

vector<int> dep(n);
const int LOG = __lg(n) + 1;
vector<vector<int>> up(n, vector<int>(LOG));

function<void(int, int, int)> dfs = [&](int u, int p, int d) {
    dep[u] = d;
    up[u][0] = p;
    for (int j = 1; j < LOG; j++) up[u][j] = up[up[u][j - 1]][j - 1];
    for (int v : tree[u]) {
        if (v != p) dfs(v, u, d + 1);
    }
};

dfs(0, 0, 0);

auto anc = [&](int u, int k) -> int {
    if (k > dep[u]) return -1;
    for (int j = 0; j < LOG; j++) {
        if (k >> j & 1) u = up[u][j];
    }
    return u;
};

auto lca = [&](int u, int v) -> int {
    if (dep[u] < dep[v]) swap(u, v);
    u = anc(u, dep[u] - dep[v]);
    if (u == v) return u;
    for (int j = LOG - 1; j >= 0; j--) {
        if (up[u][j] != up[v][j]) u = up[u][j], v = up[v][j];
    }
    return up[u][0];
};

int time = -1;
vector<int> tin(n), tout(n), euler;

function<void(int, int)> dfs = [&](int u, int p) {
    tin[u] = ++time;
    euler.push_back(u);
    for (int v : g[u]) {
        if (v != p) {
            dfs(v, u);
            euler.push_back(u);
        }
    }
    tout[u] = ++time;
};

auto kruskals_mst = [&]() -> int {
    sort(begin(edges), end(edges), [](auto e1, auto e2) {return e1[2] < e2[2];});
    DSU dsu(n);
    int res = 0, cnt = 0;
    for (auto e : edges) {
        if (!dsu.same(e[0], e[1])) {
            dsu.unite(e[0], e[1]);
            cnt++;
            res += e[2];
        }
    }
    return (cnt == n - 1 ? res : -1);
};

auto prims_mst = [&]() -> int {
    vector<int> key(n, INF);
    key[0] = 0;
    vector<bool> vis(n);
    int res = 0;
    for (int _ = 0; _ < n; _++) {
        int u = -1;
        for (int i = 0; i < n; i++) {
            if (!vis[i] and (u == -1 or key[i] < key[u])) u = i;
        }
        if (key[u] == INF) return -1;
        vis[u] = true;
        res += key[u];
        for (int v = 0; v < n; v++) {
            if (adj[u][v] < key[v]) key[v] = adj[u][v];
        }
    }
    return res;
};

/* = = = = = = =
Data Structures
= = = = = = = */

struct SegTree {
    int n;
    vector<int> st;

    int merge(int x, int y) {}

    int lc(int u) {
        return u * 2 + 1;
    }

    int rc(int u) {
        return u * 2 + 2;
    }

    SegTree(vector<int> &a) : n(size(a)), st(n * 4) {
        build(0, 0, n - 1, a);
    }

    void build(int u, int l, int r, vector<int> &a) {
        if (l == r) {
            st[u] = a[l];
            return;
        }
        int m = (l + r) / 2;
        build(lc(u), l, m, a);
        build(rc(u), m + 1, r, a);
        st[u] = merge(st[lc(u)], st[rc(u)]);
    }

    void upd(int i, int x) {
        upd(0, 0, n - 1, i, x);
    }

    void upd(int u, int l, int r, int i, int x) {
        if (l == r) {
            st[u] = x;
            return;
        }
        int m = (l + r) / 2;
        if (i <= m) upd(lc(u), l, m, i, x);
        else upd(rc(u), m + 1, r, i, x);
        st[u] = merge(st[lc(u)], st[rc(u)]);
    }

    int query(int ql, int qr) {
        return query(0, 0, n - 1, ql, qr);
    }

    int query(int u, int l, int r, int ql, int qr) {
        if (qr < l or r < ql) return 0;
        if (ql <= l and r <= qr) return st[u];
        int m = (l + r) / 2;
        return merge(query(lc(u), l, m, ql, qr), query(rc(u), m + 1, r, ql, qr));
    }
};

struct LazySegTree {
    int n;
    vector<int> st, lazy;

    int merge(int x, int y) {}

    int lc(int u) {
        return u * 2 + 1;
    }

    int rc(int u) {
        return u * 2 + 2;
    }

    void push(int u, int l, int r) {}

    LazySegTree(vector<int> &a) : n(size(a)), st(n * 4), lazy(n * 4, -1) {
        build(0, 0, n - 1, a);
    }

    void build(int u, int l, int r, vector<int> &a) {
        if (l == r) {
            st[u] = a[l];
            return;
        }
        int m = (l + r) / 2;
        build(lc(u), l, m, a);
        build(rc(u), m + 1, r, a);
        st[u] = merge(st[lc(u)], st[rc(u)]);
    }

    void upd(int ql, int qr, int val) {
        upd(0, 0, n - 1, ql, qr, val);
    }

    void upd(int u, int l, int r, int ql, int qr, int val) {
        push(u, l, r);
        if (qr < l or r < ql) return;
        if (ql <= l and r <= qr) {
            lazy[u] += val;
            push(u, l, r);
            return;
        }
        int m = (l + r) / 2;
        upd(lc(u), l, m, ql, qr, val);
        upd(rc(u), m + 1, r, ql, qr, val);
        st[u] = merge(st[lc(u)], st[rc(u)]);
    }

    int query(int ql, int qr) {
        return query(0, 0, n - 1, ql, qr);
    }

    int query(int u, int l, int r, int ql, int qr) {
        push(u, l, r);
        int m = (l + r) / 2;
        if (qr < l or r < ql) return 0;
        if (ql <= l and r <= qr) return st[u];
        return merge(query(lc(u), l, m, ql, qr), query(rc(u), m + 1, r, ql, qr));
    }
};

struct FenwickTree {
    int n;
    vector<int> bit;

    FenwickTree(vector<int> &a) : n(size(a)), bit(n + 1) {
        for (int i = 0; i < n; i++) upd(i, a[i]);
    }

    void upd(int i, int del) {
        for (i++; i <= n; i += i & -i) bit[i] += del;
    }

    int sum(int i) {
        int s = 0;
        for (i++; i > 0; i -= i & -i) s += bit[i];
        return s;
    }

    int query(int l, int r) {
        return sum(r) - sum(l - 1);
    }
};

struct FenwickTree2D {
    int n, m;
    vector<vector<int>> bit;

    FenwickTree2D(vector<vector<int>> &a) : n(size(a)), m(size(a[0])), bit(n + 1, vector<int>(m + 1)) {
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                upd(i, j, a[i][j]);
            }
        }
    }

    void upd(int r, int c, int del) {
        for (int i = r + 1; i <= n; i += i & -i) {
            for (int j = c + 1; j <= m; j += j & -j) {
                bit[i][j] += del;
            }
        }
    }

    int sum(int r, int c) {
        int s = 0;
        for (int i = r + 1; i > 0; i -= i & -i) {
            for (int j = c + 1; j > 0; j -= j & -j) {
                s += bit[i][j];
            }
        }
        return s;
    }

    int query(int r1, int c1, int r2, int c2) {
        return sum(r2, c2) - sum(r1 - 1, c2) - sum(r2, c1 - 1) + sum(r1 - 1, c1 - 1);
    }
};

struct SparseTable {
    int n, LOG;
    vector<vector<int>> table;

    int merge(int x, int y) {}

    SparseTable(vector<int> &a) : n(size(a)), LOG(__lg(n)), table(n, vector<int>(LOG + 1, -1)) {
        for (int k = 0; k <= LOG; k++) {
            for (int i = 0; i <= n - (1 << k); i++) {
                if (k == 0) table[i][0] = a[i];
                else table[i][k] = merge(table[i][k - 1], table[i + (1 << (k - 1))][k - 1]);
            }
        }
    }

    int query(int l, int r) {
        int k = __lg(r - l + 1);
        return merge(table[l][k], table[r - (1 << k) + 1][k]);
    }
};

struct SparseTable2D {
    int n, m, LOGN, LOGM;
    vector<vector<vector<vector<int>>>> table;

    int merge(int a, int b) {
        return max(a, b);
    }

    int merge(int a, int b, int c, int d) {
        return max({a, b, c, d});
    }

    SparseTable2D(vector<vector<int>> &a) :
        n(size(a)),
        m(size(a[0])),
        LOGN(__lg(n)),
        LOGM(__lg(m)),
        table(n, vector<vector<vector<int>>>(LOGN + 1, vector<vector<int>>(m, vector<int>(LOGM + 1, -1))))
    {
        for (int kn = 0; kn <= LOGN; kn++) {
            for (int i = 0; i <= n - (1 << kn); i++) {
                for (int km = 0; km <= LOGM; km++) {
                    for (int j = 0; j <= m - (1 << km); j++) {
                        if (kn == 0) {
                            if (km == 0) table[i][0][j][0] = a[i][j];
                            else table[i][0][j][km] = merge(table[i][0][j][km - 1], table[i][0][j + (1 << (km - 1))][km - 1]);
                        } else {
                            table[i][kn][j][km] = merge(table[i][kn - 1][j][km], table[i + (1 << (kn - 1))][kn - 1][j][km]);
                        }
                    }
                }
            }
        }
    }

    int query(int r1, int c1, int r2, int c2) {
        int kn = __lg(r2 - r1 + 1);
        int km = __lg(c2 - c1 + 1);
        return merge(table[r1][kn][c1][km],
                     table[r1][kn][c2 - (1 << km) + 1][km],
                     table[r2 - (1 << kn) + 1][kn][c1][km],
                     table[r2 - (1 << kn) + 1][kn][c2 - (1 << km) + 1][km]
                    );
    }
};

struct DSU {
    vi par, size;
    int comps;
    stack<int> st;

    DSU(int n) : par(n), size(n, 1), comps(n) {
        iota(begin(par), end(par), 0);
    }

    void unite(int u, int v) {
        u = find(u), v = find(v);
        if (u == v) return;
        if (size[u] < size[v]) swap(u, v);
        st.push(v);
        par[v] = u;
        size[u] += size[v];
        comps--;
    }

    int find(int u) {
        return (par[u] == u ? u : find(par[u]));
    }

    bool same(int u, int v) {
        return find(u) == find(v);
    }

    void roll() {
        int u = st.top();
        st.pop();
        par[u] = u;
        comps++;
    }
};

struct LinkedList {
    struct Node {
        int data;
        Node *next;
        Node(int d) {
            data = d;
            next = nullptr;
        }
    };

    Node *head, *tail;

    LinkedList() : head(nullptr), tail(nullptr) {}

    void append(int d) {
        Node *new_node = new Node(d);
        if (head == nullptr) {
            head = tail = new_node;
            return;
        }
        tail->next = new_node;
        tail = new_node;
    }

    void prepend(int d) {
        Node *new_node = new Node(d);
        if (head == nullptr) {
            head = tail = new_node;
            return;
        }
        new_node->next = head;
        head = new_node;
    }

    void print() {
        Node *curr_node = head;
        while (curr_node != nullptr) {
            cout << curr_node->data << "->";
            curr_node = curr_node->next;
        }
        cout << "NULL\n";
    }
};

/* = = = = = = = = = =
Square Root Techniques
= = = = = = = = = = */

const int BLOCK = 300;

vector<int> b(BLOCK);
for (int i = 0; i < n; i++) b[i / BLOCK] += a[i];

auto query = [&](int l, int r) -> int {
    int res = 0;
    int i = l;
    while (i <= r) {
        if (i % BLOCK == 0 and i + BLOCK <= r) {
            res += b[i / BLOCK];
            i += BLOCK;
        } else {
            res += a[i++];
        }
    }
    return res;
};

struct Query {
    int l, r, idx;
    bool operator<(Query q2) const {
        int b1 = l / BLOCK, b2 = q2.l / BLOCK;
        return make_pair(b1, (b1 % 2 == 0 ? r : -r)) < make_pair(b2, (b2 % 2 == 0 ? q2.r : -q2.r));
    }
};

vector<Query> qs(q);
for (int i = 0; i < q; i++) {
    Query &q = qs[i];
    cin >> q.l >> q.r;
    q.l--, q.r--;
    q.idx = i;
}

auto mos = [&]() -> vector<int> {
    sort(begin(qs), end(qs));
    auto add = [&](int i) -> void {};
    auto del = [&](int i) -> void {};
    auto get_ans = [&]() -> int {};
    vector<int> ans(size(qs));
    int l = 0, r = -1;
    for (Query q : qs) {
        while (l > q.l) add(--l);
        while (r < q.r) add(++r);
        while (l < q.l) del(l++);
        while (r > q.r) del(r--);
        ans[q.idx] = get_ans();
    }
    return ans;
};


/* = = = = = =
Stacks & Queues
= = = = = = */

vector<int> next_greater_element(vector<int> &a) {
    int n = size(a);
    vector<int> nge(n);
    stack<int> st;
    for (int i = n - 1; i >= 0; i--) {
        while (!empty(st) and a[st.top()] <= a[i]) st.pop();
        nge[i] = (empty(st) ? -1 : st.top());
        st.push(i);
    }
    return nge;
}

vector<int> sliding_window_min(vector<int> &a, int k) {
    deque<int> dq;
    vector<int> mins;
    for (int i = 0; i < size(a); i++) {
        while (!empty(dq) and dq.front() <= i - k) dq.pop_front();
        while (!empty(dq) and a[dq.back()] >= a[i]) dq.pop_back();
        dq.push_back(i);
        if (i >= k - 1) mins.push_back(a[dq.front()]);
    }
    return mins;
}
