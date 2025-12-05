import type { ModelProvider, UnifiedModel } from '../types.js';

export class PoeProvider implements ModelProvider {
  name = 'poe';
  private readonly POE_API_URL = 'https://api.poe.com/v1/models';

  async fetchModels(): Promise<UnifiedModel[]> {
    try {
      console.log(`🔍 Fetching Poe models from ${this.POE_API_URL}...`);

      const response = await fetch(this.POE_API_URL, {
        method: 'GET',
        headers: {
          'Accept': 'application/json',
          'Content-Type': 'application/json',
        },
      });

      if (!response.ok) {
        throw new Error(`Poe API request failed: ${response.status} ${response.statusText}`);
      }

      const data = await response.json();

      if (!data || !Array.isArray(data.data)) {
        throw new Error('Invalid response format from Poe API');
      }

      console.log(`📊 Received ${data.data.length} models from Poe API`);

      const models = data.data
        .filter((m: any) => m.is_available !== false)
        .map((m: any, index: number) => ({
          id: `poe/${m.id}`,
          name: m.metadata?.display_name || m.id,
          description: m.description || `${m.owned_by} model`,
          provider: 'poe' as const,
          context_length: m.context_window?.context_length ||
                          m.premium_context_limit ||
                          m.context_length ||
                          4096,
          pricing: {
            prompt: m.pricing?.prompt || '0',
            completion: m.pricing?.completion || '0'
          },
          priority: 100 + index, // Start Poe models with priority 100+
        }));

      console.log(`✅ Processed ${models.length} available Poe models`);
      return models;

    } catch (error) {
      console.error('❌ Error fetching Poe models:', error);
      throw error;
    }
  }

  transformModel(model: UnifiedModel): UnifiedModel {
    // No transformation needed for Poe models - they're already in the right format
    return model;
  }
}