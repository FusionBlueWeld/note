# 熱伝導方程式を利用して、レーザーが金属に照射された時の、金属内の温度の時間変化を表現する

熱伝導方程式を使って、レーザーが金属に照射されたときの金属内の温度の時間変化を表現することができます。
熱伝導方程式は以下のように表されます。

∂T / ∂t = α * ∇^2T
$\frac{\partial T}{\partial t} = \alpha \nabla^2 T$

ここで、
T = 温度
t = 時間
α = 熱拡散率 (熱伝導率 k / (密度 ρ * 比熱容量 c_p))
∇^2 = ラプラシアン演算子 (3次元空間では、∇^2 = ∂^2 / ∂x^2 + ∂^2 / ∂y^2 + ∂^2 / ∂z^2)

レーザー溶接の場合、レーザー照射による熱源項が熱伝導方程式に追加されます。これは、レーザー照射による金属の吸収率とレーザーのパワー分布を考慮してモデル化されます。吸収率は波長や金属の種類によって異なります。

熱伝導方程式に熱源項を追加した方程式は以下のようになります。

∂T / ∂t = α * ∇^2T + Q(x, y, z, t)
$\frac{\partial T}{\partial t} = \alpha \nabla^2 T + Q(x, y, z, t)$

ここで、
Q(x, y, z, t) = レーザー照射による熱源項

熱伝導方程式は偏微分方程式であり、解析的に解くことが難しい場合があります。そのため、数値解法（有限差分法、有限要素法、有限体積法など）を用いて解くことが一般的です。また、熱伝導方程式を解く際には、初期条件や境界条件を設定する必要があります。これらの条件は、実験やシミュレーションの目的に応じて適切に設定されます。

# レーザーが金属の表面に照射され、任意の時間の後で、金属内部で等しい温度となる点を線で結ぶ
金属内部で等しい温度となる点を線で結ぶ場合、それは等温線 (isotherm) と呼ばれます。レーザーが金属表面に照射されたときの等温線の形状は、熱伝導方程式の解によって決まります。レーザー照射による熱伝導方程式の解析的解は一般的に得られませんが、特定の条件下では単純化して近似的な解を求めることができます。

例えば、レーザーのパワー分布がガウス分布であり、照射時間が短く、熱拡散が限定的な範囲に留まる場合、熱源項は次のように表現できます。

Q(x, y, z, t) = q * exp(-((x - x0)^2 + (y - y0)^2) / (2 * σ^2)) * Θ(t)
$Q(x, y, z, t) = q \cdot \exp\left(-\frac{(x - x_0)^2 + (y - y_0)^2}{2 \sigma^2}\right) \cdot \Theta(t)$

ここで、
q = レーザーパワーのピーク値
σ = ガウス分布の標準偏差
x0, y0 = ガウス分布の中心座標
Θ(t) = ヘビサイドの階段関数 (t > 0 のとき 1、t < 0 のとき 0)

この場合、等温線は通常、照射された金属表面の近くで楕円形状を取ります。しかし、時間が経つにつれて、等温線は金属内部へ拡散し、形状は複雑になります。また、金属の熱物性や表面の処理状態、レーザーの照射条件（パワー、波長、照射時間など）によっても等温線の形状が変わります。

等温線の形状を解析的に表現することは難しいため、通常は数値解析手法（有限差分法、有限要素法、有限体積法など）を用いて熱伝導方程式を解き、等温線を計算します。計算結果から等温線の形状を抽出し、それに基づいて解析や最適化を行うことができます。

# 等温線が楕円形状になる過程を、熱伝導方程式から導出
熱伝導方程式から直接楕円形状の等温線を導出することは難しいですが、簡単な近似を使って導出を試みることができます。ここでは、短時間レーザー照射を考慮し、一次元の熱伝導方程式に焦点を当てます。

まず、1次元の熱伝導方程式を考えます。

∂T / ∂t = α * ∂^2T / ∂x^2 + Q(x, t)
$\frac{\partial T}{\partial t} = \alpha \frac{\partial^2 T}{\partial x^2} + Q(x, t)$

次に、短時間レーザー照射とガウス分布の熱源項を仮定します。

Q(x, t) = q * exp(-(x - x0)^2 / (2 * σ^2)) * Θ(t)
$Q(x, t) = q \cdot \exp\left(-\frac{(x - x_0)^2}{2 \sigma^2}\right) \cdot \Theta(t)$

ここで、Θ(t) はヘビサイドの階段関数ですが、短時間照射を考慮して、熱源項は瞬間的に発生するものとします。そのため、Θ(t) の代わりにデルタ関数 δ(t) を使います。

Q(x, t) = q * exp(-(x - x0)^2 / (2 * σ^2)) * δ(t)
$Q(x, t) = q \cdot \exp\left(-\frac{(x - x_0)^2}{2 \sigma^2}\right) \cdot \delta(t)$

この状況での熱伝導方程式は以下のようになります。

∂T / ∂t = α * ∂^2T / ∂x^2 + q * exp(-(x - x0)^2 / (2 * σ^2)) * δ(t)
$\frac{\partial T}{\partial t} = \alpha \frac{\partial^2 T}{\partial x^2} + q \cdot \exp\left(-\frac{(x - x_0)^2}{2 \sigma^2}\right) \cdot \delta(t)$

この方程式の解は、初期条件 T(x, 0) = 0 および適切な境界条件のもとで求められます。解析的な解を求めるのは難しいため、次のような近似解を試みます。

レーザー照射後の短時間内に、温度分布がガウス分布に近似できると仮定します。この仮定のもとで、温度分布 T(x, t) を次の形式で表現します。

T(x, t) = A(t) * exp(-(x - x0)^2 / (2 * σ_t^2(t)))

$T(x, t) = A(t) \cdot \exp\left(-\frac{(x - x_0)^2}{2 \sigma_t^2(t)}\right)$

ここで、A(t) は時間依存の振幅であり、σ_t(t) は時間依存のガウス分布の幅です。

温度分布 T(x, t) を熱伝導方程式に代入し、A(t) と σ_t^2(t) に関する微分方程式を導出します。

微分方程式を解くことで、A(t) と σ_t^2(t) の時間変化を求めます。

等温線は T(x, t) = 定数 の条件を満たす点の集合であるため、等温線は T(x, t) = 定数 の条件を満たす点の集合であるため、温度分布 T(x, t) についてこの条件を適用します。

T(x, t) = A(t) * exp(-(x - x0)^2 / (2 * σ_t^2(t))) = C
$T(x, t) = A(t) \cdot \exp\left(-\frac{(x - x_0)^2}{2 \sigma_t^2(t)}\right) = C$

ここで、C は定数です。

上記の式を (x - x0)^2 について解きます。

(x - x0)^2 = 2 * σ_t^2(t) * (ln(A(t)) - ln(C))
$(x - x_0)^2 = 2 \cdot \sigma_t^2(t) \cdot (\ln(A(t)) - \ln(C))$

σ_t(t) が時間とともに増加すると仮定すると、等温線の形状は楕円形状に近似されます。ただし、楕円の幅は時間とともに広がっていくことに注意してください。

(x - x0)^2 = 2 * σ_t^2(t) * (ln(A(t)) - ln(C))
$(x - x_0)^2 = 2 \cdot \sigma_t^2(t) \cdot (\ln(A(t)) - \ln(C))$

この結果は、短時間レーザー照射とガウス分布の熱源項を仮定した場合の近似解です。熱拡散が限定的な範囲に留まる場合や、2次元や3次元の状況では、等温線の形状はもっと複雑になることがあります。また、金属の熱物性や表面の処理状態、レーザーの照射条件（パワー、波長、照射時間など）によっても等温線の形状が変わります。

実際には、熱伝導方程式を解く際には数値解析手法（有限差分法、有限要素法、有限体積法など）を用いることが一般的です。数値解析により得られた温度分布から等温線の形状を計算し、それに基づいて解析や最適化を行うことができます。
