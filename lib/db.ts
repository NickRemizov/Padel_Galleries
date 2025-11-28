const postgres = require("postgres")

if (!process.env.POSTGRES_URL) {
  throw new Error("POSTGRES_URL environment variable is not defined")
}

const isSSLRequired = process.env.POSTGRES_SSL === "require"

const sql = postgres(process.env.POSTGRES_URL, {
  ssl: isSSLRequired ? { rejectUnauthorized: false } : false,
})

export { sql }
export default sql
