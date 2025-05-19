import { neon } from "@neondatabase/serverless";
import { drizzle } from "drizzle-orm/neon-http";
import * as schema from "./schema";
const sql = neon(
  "postgresql://neondb_owner:npg_IdWM6EaYsr4U@ep-plain-shadow-a4a71j1n-pooler.us-east-1.aws.neon.tech/beat_cancer?sslmode=require"
);
export const db = drizzle(sql, { schema });
