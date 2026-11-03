# Syria Fuel Tracker (2010–2026) — متتبّع أسعار المحروقات في سوريا

**[English](#english) · [العربية](#العربية)**

---

## English

### Goal
An open, documented study of Syrian fuel price decisions between 2010 and 2026: a decisions table with a source for every row, a purchasing-power analysis (litres per minimum wage), and probabilistic forecasts that are evaluated honestly.

### Status
**All current rows are leads taken from search snippets and still need re-checking against a primary source.** No number here should be relied on until its row is `verified`.

### How it works
Fuel prices in Syria are administrative: they change only when the government issues a decision. So the model does not predict the price every day. It predicts the next decision: how big the jump is and how many decisions land each month.

1. Every price is converted to new lira and to USD (the lira keeps losing value).
2. Two consecutive decisions of the same product form a *pair*; the target is the log change of the price.
3. `gap` = how far the price sits below its usual USD level. The government tends to correct part of it (`kappa`, capped at 0.5).
4. A Monte Carlo simulation (4,000 futures, Poisson number of decisions per month) gives a P10 / P50 / P90 fan for 6 months.
5. The fan is tested month by month on past data (rolling origin, no look-ahead) and calibrated.

### Results so far (output of `python fuel_forecast.py`)
| Check | Result |
|---|---|
| Fan coverage (target 80%) | **61%** over 153 forecasts from 11 monthly origins |
| Coverage by horizon | 76% at 1 month, down to 44% at 6 months |
| Real price above P90 (target 10%) | 35% |
| Backtest | 13 tests; no model clearly beats "price unchanged"; typical error 15–17% for all |
| Driver t-statistics | gap 1.08, FX −0.62, Brent 0.79 (none above 2) |

The model is not yet better than simple baselines. The calibration widens the upper band but did not raise coverage on the same history. Read the forecast as a wide range, not a prediction.

### Run it
```
pip install numpy pandas scikit-learn
python fuel_forecast.py          # full pipeline, rewrites the data block of fuel_forecast.html
python test_fuel_forecast.py     # small checks of calibration and scoring
```
`fuel_forecast.html` is the dashboard; `syria_fuel_forecast.ipynb` calls the same functions.

### Files
| Path | What |
|---|---|
| `fuel_forecast.py` | data loading, pairs, backtest, forecast, evaluation, calibration, export (commented) |
| `test_fuel_forecast.py` | checks of calibration and scoring logic |
| `fuel_forecast.html` | dashboard with the latest results |
| `data/` | decisions, FX estimates, Brent, external events |
| `media/` | motion graphics built from the dashboard data |

### Columns (data/fuel_decisions.csv)
id, announcement_date, effective_date, product, tier, price_before, price_after, price_unit, fuel_fx_per_usd, source_1..3, n_sources, status, notes

### Status policy
- `verified`: at least 2 non-aggregator sources + price before and after.
- `partial`: at least 2 sources but incomplete.
- `unverified`: one source or fewer.
- Aggregators (sp-today, globalpetrolprices) do not count.

### Known gaps
2013–2020 is almost empty; several price tiers; different exchange-rate regimes (official / parallel / fuel rate); the 100:1 currency redenomination; the gap between announcement and effective date.

### Method decision: exchange rate
USD conversions use the **parallel-market rate** (data/fx_chart_estimates.csv, a chart estimate, not a quote). The official fuel exchange rate stays a secondary column for comparison. Old lira is divided by 100 before any division. An aggregator alone means `unverified` until a press source matches the same date.

### Ethics
Published as categories, not exact dates to bet on, in a neutral tone with a disclaimer. This is not financial advice.

### Roadmap
1. Fill 2013–2020. 2. FX, Brent and wage series. 3. A news-extraction tool with precision/recall. 4. Purchasing-power analysis. 5. Probabilistic forecast beyond 2025 (Brier vs baseline, walk-forward).

### License
TODO: choose a license for data and code after checking the sources' licenses.

---

## العربية

### الهدف
دراسة مفتوحة وموثّقة لقرارات أسعار المحروقات في سوريا بين 2010 و2026: جدول قرارات بمصدر لكل صف، ثم تحليل القدرة الشرائية (لترات لكل حدّ أدنى للأجور) ونماذج احتمالية مقيّمة بصدق.

### الحالة
**كل الصفوف الحالية «خيوط» من مقتطفات بحث وتحتاج إعادة تحقق من مصدر أولي.** لا رقم هنا يُعتمد قبل أن يصير `verified`.

### كيف يعمل
أسعار المحروقات في سوريا إدارية: لا تتغير إلا عندما تصدر الحكومة قراراً. لذلك لا يتوقع النموذج السعر كل يوم، بل يتوقع القرار القادم: حجم القفزة وعدد القرارات كل شهر.

1. كل سعر يُحوَّل إلى الليرة الجديدة وإلى الدولار (الليرة تفقد قيمتها باستمرار).
2. كل قرارين متتاليين لنفس المنتج يشكّلان «زوجاً»، والهدف هو لوغاريتم تغيّر السعر.
3. `gap` = كم يقع السعر تحت مستواه المعتاد بالدولار. تميل الحكومة إلى تصحيح جزء منه (`kappa`، بسقف 0.5).
4. محاكاة مونتي كارلو (4000 مستقبل، وعدد القرارات Poisson شهرياً) تعطي مروحة P10 / P50 / P90 لستة أشهر.
5. تُختبر المروحة شهراً بشهر على بيانات الماضي (rolling origin بلا نظر للمستقبل) ثم تُعايَر.

### النتائج حتى الآن (مخرجات `python fuel_forecast.py`)
| الاختبار | النتيجة |
|---|---|
| تغطية المروحة (الهدف 80%) | **61%** على 153 توقعاً من 11 نقطة شهرية |
| التغطية حسب الأفق | 76% عند شهر، وتنزل إلى 44% عند ستة أشهر |
| السعر الحقيقي فوق P90 (الهدف 10%) | 35% |
| الاختبار الرجعي | 13 اختباراً؛ لا نموذج يتفوق بوضوح على «السعر لا يتغير»؛ الخطأ النموذجي 15–17% للجميع |
| إحصائية t للعوامل | gap 1.08، الصرف −0.62، برنت 0.79 (لا واحد فوق 2) |

النموذج لم يتفوق بعد على الخطوط الأساسية البسيطة. المعايرة توسّع الحد الأعلى لكنها لم ترفع التغطية على نفس التاريخ. اقرأ التوقع كنطاق واسع، لا كتنبؤ.

### التشغيل
```
pip install numpy pandas scikit-learn
python fuel_forecast.py          # الخط كامل، ويحدّث كتلة البيانات في fuel_forecast.html
python test_fuel_forecast.py     # فحوص صغيرة للمعايرة والتقييم
```
`fuel_forecast.html` هي لوحة النتائج، و`syria_fuel_forecast.ipynb` يستدعي نفس الدوال.

### الأعمدة (data/fuel_decisions.csv)
id, announcement_date, effective_date, product, tier, price_before, price_after, price_unit, fuel_fx_per_usd, source_1..3, n_sources, status, notes

### سياسة الحالة
- `verified`: ≥2 مصدرين غير مجمِّعين + سعر قبل وبعد.
- `partial`: ≥2 مصدر لكن ناقص.
- `unverified`: مصدر واحد أو أقل.
- المجمِّعات (sp-today، globalpetrolprices) لا تُحسب.

### الفجوات المعروفة
2013–2020 شبه غائبة؛ عدة شرائح سعرية؛ أنظمة صرف مختلفة (رسمي/موازٍ/صرف المحروقات)؛ إعادة تسمية العملة 100:1؛ فرق بين تاريخ الإعلان والتنفيذ.

### قرار منهجي: سعر الصرف
كل التحويلات للدولار تعتمد **سعر السوق الموازي** (data/fx_chart_estimates.csv، تقدير من رسم بياني وليس عرض سعر). سعر صرف المحروقات الرسمي يبقى عمودًا ثانويًا للمقارنة فقط. الليرة القديمة تُقسم على 100 قبل أي قسمة. مصدر مجمِّع وحده = `unverified` حتى يطابقه مصدر صحفي بنفس التاريخ.

### الأخلاقيات
النشر كفئات لا كتواريخ دقيقة للمراهنة، بنبرة محايدة وإخلاء مسؤولية؛ ليست نصيحة مالية.

### خارطة الطريق
1. ملء 2013–2020. 2. سلاسل الصرف وبرنت والأجور. 3. أداة استخراج من الأخبار مع precision/recall. 4. تحليل القدرة الشرائية. 5. تنبؤ احتمالي لما بعد 2025 (Brier مقابل خط أساس، walk-forward).

### الرخصة
TODO: تحديد رخصة البيانات والكود بعد فحص رخص المصادر.
