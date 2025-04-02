
import { Page, BrowserContext, Stagehand } from "@browserbasehq/stagehand";
import { z } from "zod";
import chalk from "chalk";
import dotenv from "dotenv";
import { clearOverlays, drawObserveOverlay } from "./utils.js";
import { CLOUD_SERVICES } from "./services.config.js";
import { CREDENTIALS } from "./credentials.config.js";

dotenv.config();

if (!CREDENTIALS.google.email || !CREDENTIALS.google.password) {
  throw new Error('Google credentials not set in .env file');
}

export async function main({
  page,
  context,
  stagehand,
}: {
  page: Page; 
  context: BrowserContext; 
  stagehand: Stagehand; 
}) {
  async function actWithCache(instruction: string) {
    if (instruction.includes('Google')) {
      await page.waitForTimeout(2000);
    }

    const results = await page.observe({
      instruction,
      onlyVisible: false, 
      returnAction: true, 
    });
    
    if (results.length === 0) {
      console.log(chalk.yellow(`No elements found for instruction: ${instruction}`));
      return;
    }
    
    console.log(chalk.blue("Got results:"), results);

    const actionToCache = results[0];
    console.log(chalk.blue("Taking cacheable action:"), actionToCache);

    await drawObserveOverlay(page, results);
    await page.waitForTimeout(1000); 
    await clearOverlays(page);

    await page.act(actionToCache);
  }

  for (const service of CLOUD_SERVICES) {
    console.log(chalk.green(`Starting data collection for ${service.name}`));
    
    try {
      await context.setExtraHTTPHeaders({
        'Service-Name': service.name
      });
      
      await page.goto(service.url);
      
      if (service.actions) {
        for (const instruction of service.actions) {
          console.log(chalk.blue(`Executing action: ${instruction}`));
          await actWithCache(instruction);
          await page.waitForTimeout(1000);
        }
      }
      
      await page.waitForLoadState('networkidle');
      
      console.log(chalk.green(`Completed data collection for ${service.name}`));
    } catch (error) {
      console.error(chalk.red(`Error with service ${service.name}:`, error));
    }
    
    // TODO: Add delay between services
    await page.waitForTimeout(2000);
  }
  
  await stagehand.close();
}

