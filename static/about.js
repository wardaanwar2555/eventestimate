'use server';

/**
 * @fileOverview Generates a visual mood board based on a textual description of the event's desired theme and feel.
 *
 * - generateMoodBoard - A function that handles the mood board generation process.
 */

import { ai } from '@/ai/genkit';
import { z } from 'genkit';

const GenerateMoodBoardInputSchema = z.object({
  themeDescription: z
    .string()
    .describe(
      'A detailed description of the desired theme, mood, and overall aesthetics for the event.'
    ),
});

const GenerateMoodBoardOutputSchema = z.object({
  moodBoardImageUrl: z
    .string()
    .describe(
      'A data URI containing the generated mood board image, encoded as a Base64 string.'
    ),
});

export async function generateMoodBoard(input) {
  return generateMoodBoardFlow(input);
}

const prompt = ai.definePrompt({
  name: 'generateMoodBoardPrompt',
  input: { schema: GenerateMoodBoardInputSchema },
  output: { schema: GenerateMoodBoardOutputSchema },
  prompt: `You are an AI capable of generating visual mood boards for events based on textual descriptions.

  Based on the provided description, generate an image that captures the essence of the described theme and feel.

  Description: {{{themeDescription}}}
  `,
  config: {
    safetySettings: [
      {
        category: 'HARM_CATEGORY_HATE_SPEECH',
        threshold: 'BLOCK_ONLY_HIGH',
      },
      {
        category: 'HARM_CATEGORY_DANGEROUS_CONTENT',
        threshold: 'BLOCK_NONE',
      },
      {
        category: 'HARM_CATEGORY_HARASSMENT',
        threshold: 'BLOCK_MEDIUM_AND_ABOVE',
      },
      {
        category: 'HARM_CATEGORY_SEXUALLY_EXPLICIT',
        threshold: 'BLOCK_LOW_AND_ABOVE',
      },
    ],
  },
});

const generateMoodBoardFlow = ai.defineFlow(
  {
    name: 'generateMoodBoardFlow',
    inputSchema: GenerateMoodBoardInputSchema,
    outputSchema: GenerateMoodBoardOutputSchema,
  },
  async (input) => {
    const { media } = await ai.generate({
      model: 'googleai/gemini-2.0-flash-exp',
      prompt: input.themeDescription,
      config: {
        responseModalities: ['TEXT', 'IMAGE'],
      },
    });

    return { moodBoardImageUrl: media.url };
  }
);
