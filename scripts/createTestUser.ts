import { sdk } from '../server/_core/sdk';

async function main() {
  try {
    const username = 'test@example.com';
    const password = '123456';
    const result = await sdk.registerLocal(username, password);
    console.log('Registered user:', result.user.openId, 'id=', result.user.id);
  } catch (e) {
    console.error('Register failed:', e?.message ?? e);
  }
}

main();
