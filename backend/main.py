from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import supabase

app = FastAPI()

class Listing(BaseModel):
    title: str
    price: int
    description: str

supabase.init('your-supabase-url', 'your-anon-key')

@app.get('/listings')
async def get_listings():
    data, count, _ = await supabase.table('listings').select('*, users(*)')
    return {'data': data}

@app.post('/listings')
async def create_listing(listing: Listing):
    new_listing = await supabase.table('listings').insert({'title': listing.title, 'price': listing.price, 'description': listing.description})
    return {'message': 'Listing created'}

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)
