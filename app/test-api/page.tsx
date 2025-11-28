import { galleriesApi, photographersApi, locationsApi, organizersApi, peopleApi, statsApi } from "@/lib/api"

export default async function TestApiPage() {
  const results: Record<string, { success: boolean; data?: unknown; error?: string; count?: number }> = {}

  // Test Galleries API
  try {
    const galleries = await galleriesApi.getAll()
    results.galleries = { success: true, count: galleries.length, data: galleries.slice(0, 2) }
  } catch (e) {
    results.galleries = { success: false, error: String(e) }
  }

  // Test Photographers API
  try {
    const photographers = await photographersApi.getAll()
    results.photographers = { success: true, count: photographers.length, data: photographers.slice(0, 2) }
  } catch (e) {
    results.photographers = { success: false, error: String(e) }
  }

  // Test Locations API
  try {
    const locations = await locationsApi.getAll()
    results.locations = { success: true, count: locations.length, data: locations.slice(0, 2) }
  } catch (e) {
    results.locations = { success: false, error: String(e) }
  }

  // Test Organizers API
  try {
    const organizers = await organizersApi.getAll()
    results.organizers = { success: true, count: organizers.length, data: organizers.slice(0, 2) }
  } catch (e) {
    results.organizers = { success: false, error: String(e) }
  }

  // Test People API
  try {
    const people = await peopleApi.getAll()
    results.people = { success: true, count: people.length, data: people.slice(0, 2) }
  } catch (e) {
    results.people = { success: false, error: String(e) }
  }

  // Test Stats API
  try {
    const stats = await statsApi.getRecognition()
    results.stats = { success: true, data: stats }
  } catch (e) {
    results.stats = { success: false, error: String(e) }
  }

  return (
    <div className="p-8 max-w-4xl mx-auto">
      <h1 className="text-2xl font-bold mb-6">API Test Results</h1>

      {Object.entries(results).map(([name, result]) => (
        <div key={name} className="mb-4 p-4 border rounded-lg">
          <div className="flex items-center gap-2 mb-2">
            <span className={`text-xl ${result.success ? "text-green-500" : "text-red-500"}`}>
              {result.success ? "OK" : "FAIL"}
            </span>
            <span className="font-semibold capitalize">{name}</span>
            {result.count !== undefined && <span className="text-gray-500">({result.count} items)</span>}
          </div>

          {result.error && (
            <pre className="text-red-500 text-sm bg-red-50 p-2 rounded overflow-auto">{result.error}</pre>
          )}

          {result.data && (
            <details>
              <summary className="cursor-pointer text-sm text-gray-500">Show data</summary>
              <pre className="text-xs bg-gray-50 p-2 rounded overflow-auto mt-2">
                {JSON.stringify(result.data, null, 2)}
              </pre>
            </details>
          )}
        </div>
      ))}
    </div>
  )
}
