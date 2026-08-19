// Simple horizontal bar chart, built with plain divs + Tailwind widths.
// No charting library needed — keeps this dependency-free for the demo.
//
// data: [{ label: string, value: number }]
function ChartCard({ title, data, unit = "" }) {
  const maxValue = Math.max(...data.map((d) => d.value), 0.0001);

  return (
    <div className="bg-white rounded-xl shadow-lg p-6">
      {title && (
        <h3 className="text-lg font-semibold text-green-700 mb-4">{title}</h3>
      )}

      <div className="space-y-4">
        {data.map((d) => {
          const widthPct = Math.max((d.value / maxValue) * 100, 2);
          return (
            <div key={d.label}>
              <div className="flex justify-between text-sm mb-1">
                <span className="font-medium">{d.label}</span>
                <span className="text-gray-600">
                  {d.value.toFixed(2)} {unit}
                </span>
              </div>
              <div className="w-full bg-gray-100 rounded-full h-4">
                <div
                  className="bg-green-600 h-4 rounded-full transition-all duration-500"
                  style={{ width: `${widthPct}%` }}
                />
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

export default ChartCard;
