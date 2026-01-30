import { LEGACY_CONFIG_MIGRATIONS } from "./legacy.migrations.js";
import { LEGACY_CONFIG_RULES } from "./legacy.rules.js";
export function findLegacyConfigIssues(raw) {
    if (!raw || typeof raw !== "object")
        return [];
    const root = raw;
    const issues = [];
    for (const rule of LEGACY_CONFIG_RULES) {
        let cursor = root;
        for (const key of rule.path) {
            if (!cursor || typeof cursor !== "object") {
                cursor = undefined;
                break;
            }
            cursor = cursor[key];
        }
        if (cursor !== undefined && (!rule.match || rule.match(cursor, root))) {
            issues.push({ path: rule.path.join("."), message: rule.message });
        }
    }
    return issues;
}
export function applyLegacyMigrations(raw) {
    if (!raw || typeof raw !== "object")
        return { next: null, changes: [] };
    const next = structuredClone(raw);
    const changes = [];
    for (const migration of LEGACY_CONFIG_MIGRATIONS) {
        migration.apply(next, changes);
    }
    if (changes.length === 0)
        return { next: null, changes: [] };
    return { next, changes };
}
